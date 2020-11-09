

from AMmodel.conformer import ConformerTransducer
import librosa
import tensorflow as tf


if __name__ == '__main__':

    model=ConformerTransducer(dmodel=144,reduction_factor=4,vocabulary_size=1304,
                              num_blocks=16,head_size=64,num_heads=4,
                              fc_factor=0.6,embed_dim=512,embed_dropout=0
                              ,num_lstms=2,lstm_units=320,joint_dim=1024)
    model._build([1,16000,1])
    model.load_weights('./conformer-rnnt-s/model.h5')
    wav=librosa.load('test.wav',16000)[0]
    wav=wav.reshape([1,-1,1])

    out=model.greedy_without_state(wav)
    print(out)
    out = model.greedy_with_state(wav)
    print(out)

    concrete_func = model.greedy_without_state.get_concrete_function()
    tf.saved_model.save(model, './without_state', signatures=concrete_func)
    concrete_func = model.greedy_with_state.get_concrete_function()
    tf.saved_model.save(model, './with_state', signatures=concrete_func)