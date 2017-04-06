function interest_class=extract_class(im,boxes)
    box_cell=boxes;
    row=size(im,1);
    coln=size(im,2);
    temp=[];
    interest_class=[];
    for i=1:size(box_cell,1)
        x=cell2mat(box_cell{i}(1,1));%the x of bbox
        y=cell2mat(box_cell{i}(1,2));%the y of bbox
        height=cell2mat(box_cell{i}(1,3));%the height of bbox
        width=cell2mat(box_cell{i}(1,4));%the width of bbox
        image=im(y-round(height/2):y+round(height/2)-1,x-round(width/2):x+round(width/2)-1,:);%extract the bbox image
        temp=padarray(image,[row-size(image,1),coln-size(image,2)],'post');
        interest_class=cat(4,interest_class,temp);
    end

