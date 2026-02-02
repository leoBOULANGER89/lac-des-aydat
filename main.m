table = readtable('Lake_Aydat.csv');

T_mat=table2array(table);
clear table;

%scatter3(T_mat(:,1), T_mat(:,2), T_mat(:,3))

L0 = T_mat(T_mat(:,3) == 0, :);
L4 = T_mat(T_mat(:,3) == -4, :);
L6 = T_mat(T_mat(:,3) == -6, :);
L8 = T_mat(T_mat(:,3) == -8, :);
L10 = T_mat(T_mat(:,3) == -10, :);
L12 = T_mat(T_mat(:,3) == -12, :);
L14 = T_mat(T_mat(:,3) == -14, :);
L15 = T_mat(T_mat(:,3) == -15, :);

CL = {L0, L4, L6, L8, L10, L12, L14, L15};

n0 = length(L0);
n4 = length(L4);
n6 = length(L6);
n8 = length(L8);
n10 = length(L10);
n12 = length(L12);
n14 = length(L14);
n15 = length(L15);

CN = [n0, n4, n6, n8, n10, n12, n14, n15];

figure;
view(3)
grid on
hold on

%%{
j=2;
for j=2:6
    for i=1:CN(j)
        d = (CL{j}(i,1)-CL{j+1}(:,1)).^2 + (CL{j}(i,2)-CL{j+1}(:,2)).^2;
        [~, idx] = min(d);
        plot3([CL{j}(i,1), CL{j+1}(idx,1)], [CL{j}(i,2), CL{j+1}(idx,2)], [CL{j}(i,3), CL{j+1}(idx,3)]);
        i
    end
end
%}







