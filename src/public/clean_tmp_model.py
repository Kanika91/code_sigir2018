import os, shutil

# keep the model parameters file with (hr=max_hr, ndcg=max_ndcg)
def clean_tmp_model_parameters(path, max_hr, max_ndcg):
    max_hr = float("%.4f" % max_hr)
    max_ndcg = float("%.4f" % max_ndcg)
    fs = os.listdir(path)
    for tmp_f in fs:
        tmp_num = tmp_f.split("#")
        if len(tmp_num) > 1:
            hr = float(tmp_num[1])
            ndcg = float(tmp_num[3])
            #print('hr:%s, ndcg:%s, file:%s' % (max_hr, max_ndcg, tmp_f))
            if hr < max_hr and ndcg < max_ndcg:
                dstfile = os.path.join(path, tmp_f)
                os.remove(dstfile)
            else:
                srcfile = os.path.join(path, tmp_f)
                dstfile = os.path.join(path, tmp_f.replace('#', ''))
                shutil.move(srcfile, dstfile)

def arse_clean_tmp_model_parameters(path, max_hr, max_ndcg):
    max_hr = float("%.4f" % max_hr)
    max_ndcg = float("%.4f" % max_ndcg)
    fs = os.listdir(path)
    for tmp_f in fs:
        tmp_num = tmp_f.split("#")
        if len(tmp_num) > 1:
            hr = float(tmp_num[1])
            ndcg = float(tmp_num[3])
            #print('hr:%s, ndcg:%s, file:%s' % (max_hr, max_ndcg, tmp_f))
            if hr < max_hr and ndcg < max_ndcg:
                dstfile = os.path.join(path, tmp_f)
                os.remove(dstfile)
            else:
                srcfile = os.path.join(path, tmp_f)
                dstfile = os.path.join(path, tmp_f.replace('#', ''))
                shutil.move(srcfile, dstfile)

