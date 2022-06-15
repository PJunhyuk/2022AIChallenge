import os
train_data="/DATA/train/images"
test_data="/DATA/test/images"
train_list=os.listdir(train_data)
label=dict()
label["날씨"]=dict()
label["실내여부"]=dict()
label["주간"]=dict()

for img_name in train_list:
    data=img_name.split("_")
    if len(data)==10:
        weather=data[-6]
        is_night=data[-5]
        is_in=data[-4]
    else:
        weather=data[-5]
        is_night=data[-4]
        is_in=data[-3]
    # print(is_in)
    if weather not in  label["날씨"].keys():
        label["날씨"][weather]=0
    if is_in not in  label["실내여부"].keys():
        label["실내여부"][is_in]=0
    if is_night not in  label["주간"].keys():
        label["주간"][is_night]=0
    label["날씨"][weather]+=1
    label["실내여부"][is_in]+=1
    label["주간"][is_night]+=1
print(label)