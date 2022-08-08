./gan/bin/python main.py --log_dir=AdvCBCT --norm=True --batch_size=16 --z_size=256 --cbct=True --niter=10000
./gan/bin/python main.py --log_dir=Adv4D --norm=True --batch_size=16 --z_size=256 --niter=10000
./gan/bin/python main.py --log_dir=AdvCBCT512 --norm=True --batch_size=16 --cbct=True
./gan/bin/python main.py --log_dir=Adv4D512 --norm=True --batch_size=16
./run.sh