import sys
import os

# method to download file

def download(path,store):
    from urllib import request
    fname = store+path.split('/')[-1]

    print('Downloading ' + path)

    def progress(count, block_size, total_size):
        if count % 20 == 0:
            print('Downloaded %02.02f/%02.02f MB' % (
                count * block_size / 1024.0 / 1024.0,
                total_size / 1024.0 / 1024.0), end='\r')

    filepath, _ = request.urlretrieve(path, filename=fname, reporthook=progress)
    
    return filepath


print ("checking requirement files....")

if os.path.exists('sequence2sequence/checkpoints/checkpoint'):
    print("checkpoints files present")
else:
    print('downloading checkpoint files')
    check_list = ['chatbot-1165000.data-00000-of-00001','chatbot-1165000.index','chatbot-1165000.meta','checkpoint']
    for item in check_list:
        url = "https://s3-us-west-1.amazonaws.com/cmpe297-checkpoint/"+item
        download(url,'sequence2sequence/checkpoints/')


if os.path.exists('sequence2sequence/processed/test.dec'):
    print("processed files present")
else:
    print('downloading processed files')
    check_list = ['test.dec','test.enc','test_ids.dec','test_ids.enc','train.dec','train.enc','train_ids.dec',
                  'train_ids.enc','vocab.dec','vocab.enc']
    for item in check_list:
        url = "https://s3-us-west-1.amazonaws.com/cmpe297-checkpoint/"+item
        download(url,'sequence2sequence/processed/')
