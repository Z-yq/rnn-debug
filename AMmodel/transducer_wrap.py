import os
import tensorflow as tf

from AMmodel.layers.time_frequency import Melspectrogram


class TransducerPrediction(tf.keras.Model):
    def __init__(self,
                 vocabulary_size: int,
                 embed_dim: int,
                 embed_dropout: float = 0,
                 num_lstms: int = 1,
                 lstm_units: int = 512,
                 name="transducer_prediction",
                 **kwargs):
        super(TransducerPrediction, self).__init__(name=name, **kwargs)
        self.embed = tf.keras.layers.Embedding(
            input_dim=vocabulary_size, output_dim=embed_dim, mask_zero=False)
        self.do = tf.keras.layers.Dropout(embed_dropout)
        self.lstm_cells = []
        # lstms units must equal (for using beam search)
        for i in range(num_lstms):
            lstm = tf.keras.layers.LSTMCell(units=lstm_units,
                                            )
            self.lstm_cells.append(lstm)
        self.decoder_lstms = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(
            self.lstm_cells, name="decoder_lstms"
        ), return_sequences=True, return_state=True)

    def get_initial_state(self, input_sample):

        return self.decoder_lstms.get_initial_state(input_sample)

    # @tf.function(experimental_relax_shapes=True)
    def call(self,
             inputs,
             training=False,
             p_memory_states=None,
             **kwargs):

        outputs = self.embed(inputs, training=training)
        outputs = self.do(outputs, training=training)
        if p_memory_states is None:  # Zeros mean no initial_state
            p_memory_states = self.get_initial_state(outputs)

        outputs = self.decoder_lstms(outputs, training=training, initial_state=p_memory_states)
        # new_memory_states = outputs[1:]
        outputs = outputs[0]

        return outputs  # , new_memory_states
    def call_states(self,
             inputs,
             training=False,
             p_memory_states=None,
             **kwargs):

        outputs = self.embed(inputs, training=training)
        outputs = self.do(outputs, training=training)
        if p_memory_states is None:  # Zeros mean no initial_state
            p_memory_states = self.get_initial_state(outputs)

        p_m=[
            [p_memory_states[0][0],p_memory_states[0][1]],
            [p_memory_states[1][0],p_memory_states[1][1]]
        ]
        outputs = self.decoder_lstms(outputs, training=training, initial_state=p_m)
        new_memory_states = tf.cast(outputs[1:],tf.float32)

        outputs = outputs[0]

        return outputs  , new_memory_states
    def get_config(self):
        conf = super(TransducerPrediction, self).get_config()
        conf.update(self.embed.get_config())
        conf.update(self.do.get_config())
        for lstm in self.lstms:
            conf.update(lstm.get_config())
        return conf


class TransducerJoint(tf.keras.Model):
    def __init__(self,
                 vocabulary_size: int,
                 joint_dim: int = 1024,
                 name="tranducer_joint",
                 **kwargs):
        super(TransducerJoint, self).__init__(name=name, **kwargs)
        self.ffn_enc = tf.keras.layers.Dense(joint_dim)
        self.ffn_pred = tf.keras.layers.Dense(joint_dim)
        self.ffn_out = tf.keras.layers.Dense(vocabulary_size)

    # @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False, **kwargs):
        # enc has shape [B, T, E]
        # pred has shape [B, U, P]
        enc, pred = inputs
        enc_out = self.ffn_enc(enc, training=training)  # [B, T ,E] => [B, T, V]
        pred_out = self.ffn_pred(pred, training=training)  # [B, U, P] => [B, U, V]
        # => [B, T, U, V]
        outputs = tf.nn.tanh(tf.expand_dims(enc_out, axis=2) + tf.expand_dims(pred_out, axis=1))
        outputs = self.ffn_out(outputs, training=training)
        return outputs

    def get_config(self):
        conf = super(TransducerJoint, self).get_config()
        conf.update(self.ffn_enc.get_config())
        conf.update(self.ffn_pred.get_config())
        conf.update(self.ffn_out.get_config())
        return conf


class Transducer(tf.keras.Model):
    """ Transducer Model Warper """

    def __init__(self,
                 encoder: tf.keras.Model,
                 vocabulary_size: int,
                 embed_dim: int = 512,
                 embed_dropout: float = 0,
                 num_lstms: int = 1,
                 lstm_units: int = 320,
                 joint_dim: int = 1024,
                 name="transducer",
                 **kwargs):
        super(Transducer, self).__init__(name=name, **kwargs)
        self.encoder = encoder
        self.predict_net = TransducerPrediction(
            vocabulary_size=vocabulary_size,
            embed_dim=embed_dim,
            embed_dropout=embed_dropout,
            num_lstms=num_lstms,
            lstm_units=lstm_units,
            name=f"{name}_prediction"
        )
        self.joint_net = TransducerJoint(
            vocabulary_size=vocabulary_size,
            joint_dim=joint_dim,
            name=f"{name}_joint"
        )

        self.mel_layer = Melspectrogram(sr=16000,
                                        n_mels=80,
                                        n_hop=int(
                                            10 * 16000 // 1000),
                                        n_dft=1024,
                                        trainable_fb=True
                                        )

        self.kept_decode = None
        self.startid = 0
        self.endid = 1
        self.max_iter = 10

    def _build(self, sample_shape):  # Call on real data for building model
        features = tf.random.normal(shape=sample_shape)
        predicted = tf.constant([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        return self([features, predicted], training=True)

    def save_seperate(self, path_to_dir: str):
        self.encoder.save(os.path.join(path_to_dir, "encoder"))
        self.predict_net.save(os.path.join(path_to_dir, "prediction"))
        self.joint_net.save(os.path.join(path_to_dir, "joint"))

    def summary(self, line_length=None, **kwargs):
        self.encoder.summary(line_length=line_length, **kwargs)
        self.predict_net.summary(line_length=line_length, **kwargs)
        self.joint_net.summary(line_length=line_length, **kwargs)
        super(Transducer, self).summary(line_length=line_length, **kwargs)

    # @tf.function(experimental_relax_shapes=True)
    def call(self,inputs, training=False):
        features, predicted=inputs

        if self.mel_layer is not None:
            features = self.mel_layer(features)
        enc = self.encoder(features, training=training)
        pred = self.predict_net(predicted, training=training)
        outputs = self.joint_net([enc, pred], training=training)

        return outputs


    @tf.function(experimental_relax_shapes=True,
                 input_signature=[tf.TensorSpec([1,None,1],tf.float32)])
    def greedy_without_state(self,
                       features
                       ):

        features = self.mel_layer(features)

        decoded = tf.constant([0])

        enc = self.encoder(features, training=False)  # [1, T, E]
        enc = tf.squeeze(enc, axis=0)  # [T, E]

        T = tf.cast(tf.shape(enc)[0], dtype=tf.int32)

        i = tf.constant(0, dtype=tf.int32)

        def _cond(enc, i, decoded, T):
            return tf.less(i, T)

        def _body(enc, i, decoded, T):
            hi = tf.reshape(enc[i], [1, 1, -1])  # [1, 1, E]
            y = self.predict_net(
                inputs=tf.reshape(decoded, [1, -1]),  # [1, 1]
                p_memory_states=None,
                training=False
            )
            y = y[:, -1:]
            # [1, 1, P], [1, P], [1, P]
            # [1, 1, E] + [1, 1, P] => [1, 1, 1, V]
            ytu = tf.nn.log_softmax(self.joint_net([hi, y], training=False))
            ytu = tf.squeeze(ytu, axis=None)  # [1, 1, 1, V] => [V]
            n_predict = tf.argmax(ytu, axis=-1, output_type=tf.int32)  # => argmax []
            n_predict = tf.reshape(n_predict, [1])

            def return_no_blank():
                return tf.concat([decoded, n_predict], axis=0)

            decoded = tf.cond(
                n_predict != 1303 and n_predict != 0,
                true_fn=return_no_blank,
                false_fn=lambda: decoded
            )

            return enc, i + 1, decoded, T

        _, _, decoded, _ = tf.while_loop(
            _cond,
            _body,
            loop_vars=(enc, i, decoded, T),
            shape_invariants=(
                tf.TensorShape([None, None]),
                tf.TensorShape([]),

                tf.TensorShape([None]),

                tf.TensorShape([])
            )
        )

        return tf.expand_dims(decoded, axis=0)

    @tf.function(experimental_relax_shapes=True,
                 input_signature=[tf.TensorSpec([1,None,1],tf.float32)])
    def greedy_with_state(self,
                             features
                             ):

        features = self.mel_layer(features)

        decoded = tf.constant([0])

        enc = self.encoder(features, training=False)  # [1, T, E]
        h = tf.cast(self.predict_net.get_initial_state(enc), tf.float32)
        enc = tf.squeeze(enc, axis=0)  # [T, E]

        T = tf.cast(tf.shape(enc)[0], dtype=tf.int32)

        i = tf.constant(0, dtype=tf.int32)

        def _cond(enc, i, decoded, h,T):
            return tf.less(i, T)

        def _body(enc, i, decoded,h, T):
            hi = tf.reshape(enc[i], [1, 1, -1])  # [1, 1, E]
            y,h_ = self.predict_net.call_states(
                inputs=tf.reshape(decoded[-1], [1, 1]),  # [1, 1]
                p_memory_states=h,
                training=False
            )
            # [1, 1, P], [1, P], [1, P]
            # [1, 1, E] + [1, 1, P] => [1, 1, 1, V]

            ytu = tf.nn.log_softmax(self.joint_net([hi, y], training=False))
            ytu = tf.squeeze(ytu, axis=None)  # [1, 1, 1, V] => [V]
            n_predict = tf.argmax(ytu, axis=-1, output_type=tf.int32)  # => argmax []
            n_predict = tf.reshape(n_predict, [1])

            def return_no_blank():
                return [tf.concat([decoded, n_predict], axis=0),h_]

            decoded ,h= tf.cond(
                n_predict != 1303 and n_predict != 0,
                true_fn=return_no_blank,
                false_fn=lambda: [decoded,h]
            )

            return enc, i + 1, decoded,h, T

        _, _, decoded,h, _ = tf.while_loop(
            _cond,
            _body,
            loop_vars=(enc, i, decoded,h, T),
            shape_invariants=(
                tf.TensorShape([None, None]),
                tf.TensorShape([]),
                tf.TensorShape([None]),
                tf.TensorShape([2,2,1,320]),
                tf.TensorShape([])
            )
        )

        return tf.expand_dims(decoded, axis=0)

    def get_config(self):
        if self.mel_layer is not None:
            conf = self.mel_layer.get_config()
            conf.update(self.encoder.get_config())
        else:
            conf = self.encoder.get_config()
        conf.update(self.predict_net.get_config())
        conf.update(self.joint_net.get_config())
        return conf
