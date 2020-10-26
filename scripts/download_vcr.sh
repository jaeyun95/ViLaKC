# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DOWNLOAD=$1

for FOLDER in 'img_db' 'txt_db'; do
    if [ ! -d $DOWNLOAD/$FOLDER ] ; then
        mkdir -p $DOWNLOAD/$FOLDER
    fi
done

BLOB='https://convaisharables.blob.core.windows.net/uniter'

# image dbs
for SPLIT in 'train' 'val' 'test' 'gt_train' 'gt_val' 'gt_test'; do
    if [ ! -d $DOWNLOAD/img_db/vcr_$SPLIT ] ; then
        wget $BLOB/img_db/vcr_$SPLIT.tar -P $DOWNLOAD/img_db/
        tar -xvf $DOWNLOAD/img_db/vcr_$SPLIT.tar -C $DOWNLOAD/img_db
    fi
done

# text dbs
for SPLIT in 'train' 'val' 'test'; do
    wget $BLOB/txt_db/vcr_$SPLIT.db.tar -P $DOWNLOAD/txt_db/
    tar -xvf $DOWNLOAD/txt_db/vcr_$SPLIT.db.tar -C $DOWNLOAD/txt_db
done
