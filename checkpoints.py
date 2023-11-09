from lm_checkpoints import PythiaCheckpoints

for ckpt in PythiaCheckpoints(step=[0], seed=[0]):
    print(ckpt)
