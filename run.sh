./gan/bin/python main.py --log_dir=Adv4DMD --norm=True --z_size=256 --batch_size=16 --md=True
./gan/bin/python main.py --log_dir=AdvCBCTMD --norm=True --z_size=256 --batch_size=16 --cbct=True --md=True
./gan/bin/python main.py --log_dir=AdvLIDCMD --norm=True --batch_size=8 --lidc=True --md=True
./run.sh