export LD_LIBRARY_PATH=/tmp/caffe/ffmpeg:/opt/cuda-8.0-cuDNN5.1/lib64:/tmp/caffe/opencv/install/lib:/tmp/caffe/anaconda2/lib:/opt/OpenBLAS/lib:/usr/local/lib
export PATH=/opt/cuda-8.0-cuDNN5.1/bin:/tmp/caffe/anaconda2/bin:/usr/lib64/qt-3.3/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin


addpath('/tmp/caffe/matlab/');
caffe.set_device(1);
caffe.set_mode_gpu();
image_mean = caffe.io.read_mean('/tmp/caffe/models/data_mean.binaryproto');
immatlagesets=imageSet('/tmp/caffe/data','recursive'); 



setmatrix={0,0,0};
count=1;
for i=3:25
   nam=dir('/afs/cs.pitt.edu/usr0/zih7/cars/');
   D=dir(['/afs/cs.pitt.edu/usr0/zih7/cars/',nam(i).name,'/*.jpg']);
   num=length(D(not([D.isdir])));
    for j=1:1:num
        setmatrix(end+1,:)={i-2,['/afs/cs.pitt.edu/usr0/zih7/cars/',nam(i).name,'/',D(j).name],count};
        count=count+1;
    end
end
setmatrix(1,:)=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
setmatrix=datasample(setmatrix,2440);
Trainmatrix=setmatrix(1:1952,:);
setmatrix(1:1952,:)=[];
Testmatrix=setmatrix(1:244,:);
setmatrix(1:244,:)=[];
Validationmatrix=setmatrix;
%%%%%initialize to train my network%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
loopmatrix=[];
validationloopmatrix=[];
Vepochlabel=[];
accuracymatrix=[];
epochlabel=[];
lossmatrix=[];
train=Trainmatrix;
validation=Validationmatrix;

%%%%epoch 1 to 25
for epoch=1:1:25
        for batch=1:1:2444  %1668 iterations
                image8=train(1:8,:);
                train(1:8,:)=[];
                for i=1:1:8 %pack 8 images together
                     
                     dir={image8(i,2)};
                     dir=dir{1};
                     temp=caffe.io.load_image(dir{1});
                     temp1=imresize(temp,[227,227])-image_mean;
                     loopmatrix=cat(4,loopmatrix,temp1);
                end
             epochlabel=image8(:,1);
             epochlabel=cell2mat(epochlabel)-ones(8,1);
             solver.net.blobs('data').set_data(loopmatrix);
             solver.net.blobs('label').set_data(epochlabel);
             loopmatrix=[];
             solver.step(1);
             solvervalue=solver.net.blobs('loss').get_data();
             lossmatrix(batch,:)=solvervalue;
        end        
        train=Trainmatrix;
        train=datasample(train,1952);
        accuracy=0;
      for Vbatch=1:1:30  %validation iterations
                    imageV8=validation(1:8,:);
                    validation(1:8,:)=[];
                for k=1:1:8
                    dir={imageV8(k,2)};
                    dir=dir{1};
                    temp2=caffe.io.load_image(dir{1});
                    temp3=imresize(temp2,[227,227])-image_mean;
                    validationloopmatrix=cat(4,validationloopmatrix,temp3);
                end
            Vepochlabel=imageV8(:,1);
            Vepochlabel=cell2mat(Vepochlabel)-ones(8,1);
            solver.net.blobs('data').set_data(validationloopmatrix);
            solver.net.blobs('label').set_data(Vepochlabel);
            validationloopmatrix=[];
            solver.net.forward_prefilled();
            % get accuracy
            accuracy=solver.net.blobs('accuracy').get_data()+accuracy;
         
      end
      validation=Validationmatrix;
      validation=datasample(validation,244);
      accuracymatrix(epoch,:)=accuracy/30;      

end
solver.net.save('yuzhi.caffemodel');


%%%%%%%%plot figures of accuracy and loss%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1)
plot(accuracymatrix);
title('acc');
xlabel=('epoch');
ylabel=('accuracy');
saveas(gcf,'acc.png');

figure(2)
plot(lossmatrix);
title('loss');
xlabel('iteration');
ylabel('loss');
saveas(gcf,'loss.png');
