_dict_mult = 32
DICT_SIZE = 2048*_dict_mult
EMBED_DIM  = 16
LATENT_DIM = 32
BATCH_SIZE = 512
NUM_EPOCHS = 16

def get_embed_dict_name():
    return "embed_"+str(_dict_mult)+"_"+str(EMBED_DIM)+".dict"