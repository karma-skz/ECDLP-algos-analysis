attach("main.sage")
attach("functions.sage")
def create_data():
    import time
    global file_data
    no_try=10
    for i in range(8,10):
        av_ntry=0;a=next_prime(ZZ.random_element(10^i))
        timestamp=time.strftime("%Y%m%d-%H%M")
        filename=("data_prime"+str(timestamp)+"."+"txt")
        file_data="/home/ayan/prime_data/"+filename
        for j in range(no_try):
            temp_try=compute_dlp(a)
            av_ntry=av_ntry+temp_try
        av_ntry=int((av_ntry)/no_try)
        out=open(file_data,'a+')
        out.write("===========================================\n")
        out.write("average ntry for "+str(no_try)+" tries was "+str(av_ntry)+"\n")
        out.write("===========================================\n")
        out.close()
