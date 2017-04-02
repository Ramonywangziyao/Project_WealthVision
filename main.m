%load instagram profiles
profile=[]
SVM_label=[]
%faster RCNN extract objects

%---------CODE HERE---------

%CNN traverse, extract label
img=load_img()
cnn_model=train_CNN()
%labels 1XN matrix
%size of profile

for i=1:size(profile,1)
    object_label(i,:)=extract_label(profile(i),cnn_model)
end

descriptors=compute_descriptors(object_label)
%train SVM
svm_model=fitcecoc(descriptors,SVM_label);

%predict
