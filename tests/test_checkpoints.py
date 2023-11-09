from lm_checkpoints import PythiaCheckpoints, MultiBERTCheckpoints

def test__pythia_sizes():
    pythia_ckpts = PythiaCheckpoints(steps=[143000], seeds=[0])
    assert len(pythia_ckpts) == 1

    pythia_ckpts = PythiaCheckpoints(steps=[0, 143000], seeds=[0,1,2,3,4])
    assert len(pythia_ckpts) == 5*2

def test__pythia_splits():
    pythia_ckpts = PythiaCheckpoints(steps=[143000], seeds=[0, 1])
    ckpts_splits = pythia_ckpts.split(2)
    assert len(ckpts_splits) == 2
    assert len(ckpts_splits[0]) == 1

def test__multiberts_sizes():
    berts_ckpts = MultiBERTCheckpoints(steps=[0], seeds=[0])
    assert len(berts_ckpts) == 1

    berts_ckpts = MultiBERTCheckpoints(steps=[0, 2000], seeds=[0,1,2,3,4])
    assert len(berts_ckpts) == 5*2

def test__multiberts_splits():
    berts_ckpts = MultiBERTCheckpoints(steps=[0], seeds=[0, 1])
    ckpts_splits = berts_ckpts.split(2)
    assert len(berts_ckpts) == 2
    assert len(ckpts_splits[0]) == 1