rlaunch \
--gpu=1 \
--memory=160000 \
--cpu=16 \
--charged-group=ai4sdata_gpu \
--private-machine=group \
--mount=gpfs://gpfs1/zhaoxiangyu:/mnt/shared-storage-user/zhaoxiangyu \
--mount=gpfs://gpfs1/sciprismax:/mnt/shared-storage-user/sciprismax -- bash