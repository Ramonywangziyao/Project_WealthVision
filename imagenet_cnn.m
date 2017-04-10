function label=imagenet_cnn(extract_objects)
%%
%initialize
%source /opt/cs2770/env.sh
% addpath('/opt/cs2770/caffe-rc3/matlab');
% caffe.set_device(1);
% caffe.set_mode_gpu();
image_mean = caffe.io.read_mean('/opt/cs2770/caffe-rc3/data/ilsvrc12/imagenet_mean.binaryproto');
image_mean=imresize(image_mean,[227,227]);
%%
%import the best pretrained model
net = caffe.Net('/afs/cs.pitt.edu/usr0/zih7/deploy.prototxt', '/opt/cs2770/caffe-rc3/models/bvlc_alexnet/bvlc_alexnet.caffemodel', 'test');

%%
%test cnn
label=[];
car_feature=[];
for i=1:size(extract_objects,4)
    object=extract_objects(:,:,:,i);
    temp1=imresize(object,[227,227])-image_mean;
    net.forward({temp1});
    temp2=net.blobs('fc8').get_data();
    car_feature=temp2';
    [~,index]=max(car_feature,[],2);
    label(end+1,:)=index-1;
%     if index-1==817
%         category='sport car';
%     elseif index-1==409
%         category='watch';
%     elseif index-1=770
%         category='shoe';
%     end
%     label(end+1,:)=category;
end



%%
%import car data
% cars={0,0,0};
% count=1;
% i=3;
%     nam=dir('/afs/cs.pitt.edu/usr0/zih7/1/');
%     D=dir(['/afs/cs.pitt.edu/usr0/zih7/1/',nam(i).name,'/*.jpg']);
%     num=length(D(not([D.isdir])));
%     for j=1:1:num
%         cars(end+1,:)={i-2,['/afs/cs.pitt.edu/usr0/zih7/1/',nam(i).name,'/',D(j).name],count};
%         count=count+1;
%     end
% 
% cars(1,:)=[];
% car_feature=[];
% for i=1:50
%     dir={cars(i,2)};
%     dir=dir{1};
%     temp=caffe.io.load_image(dir{1});
%     temp1=imresize(temp,[227,227])-image_mean;
%     net.forward({temp1});
%     temp2=net.blobs('fc8').get_data();
%     temp2=temp2';
%     car_feature(end+1,:)=temp2;
% end
% [value,index]=max(car_feature,[],2);
% a=1;
% for i=1:50
%     if index(i,:)-1==817
%         a=a+1;
%     end
% end
% acc=a/50