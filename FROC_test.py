import csv
import copy
def FROC(conf_list,neg_rate_list,sacns_num,pos_num):
    temp=copy.deepcopy(conf_list)
    temp.sort()
    temp.reverse()
    pos=0
    neg=0
    index=0
    pos_rate_list=[]
    
    for i in range(len(temp)):
        if temp[i][1]:
            pos+=1
        else:
            neg+=1
        if neg>=sacns_num*neg_rate_list[index]:
            #print 'fp/scans:',neg_rate_list[index],'accuracy:',float(pos)/pos_num
            pos_rate_list.append(float(pos)/pos_num)
            if index>=len(neg_rate_list)-1:       
                break
            else:
                index+=1
    return pos_rate_list


def FROC_seg(conf_list,neg_rate_list,sacns_num,pos_num):
    temp=copy.deepcopy(conf_list)
    temp.sort()
    temp.reverse()
    pos=0
    neg=0
    index=0
    pos_rate_list=[]
    
    for i in range(len(temp)):
        if temp[i][1]:
            pos+=1
        else:
            neg+=1
        if neg>=sacns_num*neg_rate_list[index]:
            #print 'fp/scans:',neg_rate_list[index],'accuracy:',float(pos)/pos_num
            pos_rate_list.append(float(pos)/pos_num)
            if index>=len(neg_rate_list)-1:       
                break
            else:
                index+=1
    return pos_rate_list

if __name__ == "__main__":
    csvfile = file('predication.csv', 'rb')
    reader = csv.reader(csvfile)
    
    for line in reader:
        print line
        break
    
    seriesuid_list=[]
    conf_list=[]
    pos_num=0
    for line in reader:
        seriesuid=line[0]
        conf=float(line[4])
        label=line[5]=='True'
        pos_num+=label
        seriesuid_list.append(seriesuid)
        conf_list.append([conf,label])
        
        
    sacns_num=len(set(seriesuid_list))
    
    neg_rate_list=[0.125,0.25,0.5,1,2,4,8]
    
    pos_rate_list=FROC(conf_list,neg_rate_list,sacns_num,pos_num)
        
    print sum(pos_rate_list)/len(pos_rate_list)
            
        
    
    
    csvfile.close() 