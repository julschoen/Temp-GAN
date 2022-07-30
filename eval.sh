#./gan/bin/python eval_im.py --log_dir=4D_eval -l=AdvNFNorm4d -l=AdvOne4D --data_path=../Data/4dct_clean/test_pat.npz
#./gan/bin/python eval_im.py --log_dir=LIDC_eval -l=AdvNFNorm -l=ClNFNorm -l=TrNFNorm --data_path=../3D-GAN/test_lidc_128.npz
#./gan/bin/python eval_im.py --log_dir=LIDC_eval -l=AdvNFNorm -l=ClNFNorm -l=TrNFNorm --data_path=../3D-GAN/test_lidc_128.npz
./gan/bin/python eval_temp.py --log_dir=LIDCTestTemp_eval -l=AdvTest -l=ClTest2 -l=TrTest2 -l=AdvTest2
#./gan/bin/python eval_temp.py --log_dir=4DTemp_eval -l=AdvNFNorm4d -l=AdvOne4D


