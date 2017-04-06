function descriptor,label = compute_descriptor(obj_label,class_label)
    descriptor=zeros(1,6)
    %need to add weight
    label = class_label
    for i=1:size(obj_label,1)
       if obj_label(i,1) == 'sportcar'
           descriptor(1,1) = descriptor(1,1)+1
       elseif obj_label(i,1) == 'watch'
           descriptor(1,2) = descriptor(1,2)+1
       elseif obj_label(i,1) == 'shoe'
           descriptor(1,3) = descriptor(1,3)+1
       elseif obj_label(i,1) == 'bag'
           descriptor(1,4) = descriptor(1,4)+1
       elseif obj_label(i,1) == 'logo'
           descriptor(1,5) = descriptor(1,5)+1
       else
           descriptor(1,6) = descriptor(1,6)+1
       end
    end
end
