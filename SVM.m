%% conducting grid search and tuning the hyperparameters
%Listing the hyperparameters for SVM model
load('Training.mat') %import training data 
clc;
tic
%diary allows to move the output into a text file for easy interpretation
diary grid_value12.out
rng();
%considering box constraints in the range of 0.3 to 0.6
BC = [0.3;0.4;0.5;0.6];

%SVM can be modelled using three kernel functions as listed below
kernelFunc = containers.Map({'Linear','Gaussian', 'Polynomial'},...
    {'Linear', 'Gaussian','Polynomial'});
k = keys(kernelFunc);
val = values(kernelFunc);

%creating a matrix that will collate the mean accuracy for Linear and
%Gaussian kernels as a combination with each box constraint
mymatrix =table('Size', [0, 5], ...
                      'VariableTypes', {'string', 'double', 'double', 'double','double'}, ...
                      'VariableNames', {'Kernel', 'Box','poly_order','Loss','Accuracy' ...
                      });
%creating a matrix that will collate the mean accuracy for Polynomial kernel 
%as a combination with each box constraint and polynomial order                                     });
mymatrix1 =table('Size', [0, 5], ...
                      'VariableTypes', {'string', 'double','double','double','double'}, ...
                      'VariableNames', {'Kernel', 'Box','poly_order','Loss',...
                                       'Accuracy'});
%initialising the loss value
Loss_val =[];

%initialising the polynomial order
Poly_order = [2;3;4];

%K-fold cross validation with 10 iterations for each kernal, box constraint
%and polynomial order (for polynomial function) fit into the SVM model to
%identify the most suited model
disp("START")
%initialising the iterations for kernel functions
for j = 1 :size( kernelFunc)
%initialising the iterations for box constraints
    for i = 1: size( BC)
            %assigning each value of kernelfunc for iteration
            kernel = val{j};
            %assigning each value of box constraint for iteration
            box = BC(i);
            %get the number of samples in the training set
            n=size(Training,1);
            %cvpartition using kfold cross validation
            c = cvpartition(n,'kfold',10);
            
            %Using cvpartition to implement k-fold cross validation for each kernel function,
            %box constraint and polynomial order (for polynomial kernel
            %function)
            for m = 1:10
                disp(m)
                train_idx=training(c,m);
                train_data= Training(train_idx,:);
                val_idx=test(c,m);
                val_data=Training(val_idx,:);
                x=train_data(:,1:10);
                t=train_data(:,14);
                y=val_data(:,1:10);
                target_labels=val_data(:,14);
                target_labels = target_labels{:,:};
                %loop for Linear Kernel function
                if strcmpi('Linear',kernel)
                    
                    disp(box)
                    %fitting the Linear kernel function with different box
                    %constraints into the SVM model
                    mdl = fitcsvm(x, t, 'KernelFunction', kernel, 'BoxConstraint', box)
                    fprintf("Kernel is Linear")
                    %predict the label, score and store it in a table
                    [labelL, score] = predict(mdl, y)
                    t3=table(target_labels,labelL,score);
                    t3.Properties.VariableNames = {'TrueLabel','PredictedLabel','Score'};
                    %cacluating the loss for the model
                    Loss = resubLoss(mdl)
                    %appending the loss from each iteration
                    Loss_val(m) = [Loss];
            if mod(m,10)== 0
               %calculate the mean loss
               Loss1 = mean(Loss_val)
               %since this is a linear function, there is no polynomial
               %order. For consistency of matrix columns, this has been
               %assigned as zero
               poly_order = 0
               %plot confusion matrix and calculate accuracy
               CM_mdl = confusionmat(target_labels, labelL)
               Accuracy = 100*sum(diag(CM_mdl))./sum(CM_mdl(:));
               %define the columns of the matrix
               myrow ={kernel,box, poly_order, Loss1, Accuracy}
               mymatrix = [mymatrix;myrow]
            end
            %loop for Gaussian kernel function
               elseif strcmpi('Gaussian',kernel)
                    %fitting the Gaussian kernel function with different box
                    %constraints into the SVM model
                    mdl = fitcsvm(x, t, 'KernelFunction', kernel, 'BoxConstraint', box)
                    fprintf("Kernel is gaussian")
                    %predict the label, score and store it in a table
                    [labelG, score] = predict(mdl, y)
                    t4=table(target_labels,labelG,score);
                    t4.Properties.VariableNames = {'TrueLabel','PredictedLabel','Score'};
                    %cacluating the loss for the model
                    Loss = resubLoss(mdl)
                    Loss_val(m) = [Loss];
             if mod(m,10)== 0
               %calculate the mean loss
               Loss1 = mean(Loss_val)
               %since this is a linear function, there is no polynomial
               %order. For consistency of matrix columns, this has been
               %assigned as zero
               poly_order = 0
               %plot confusion matrix and calculate accuracy
               CM_mdl = confusionmat(target_labels, labelG)
               Accuracy = 100*sum(diag(CM_mdl))./sum(CM_mdl(:));
               %define the columns of the matrix
               myrow ={kernel,box, poly_order, Loss1, Accuracy}
               mymatrix = [mymatrix;myrow]      
             end   
             %loop for Polynomial kernel function
                elseif strcmpi('Polynomial',kernel)
                    %initialise Poly_order
                    for k=1:size(Poly_order)
                    poly_order = Poly_order(k)
                    %fitting the Gaussian kernel function with different box
                    %constraints and polynomial orders into the SVM model
                    mdl = fitcsvm(x, t,  'BoxConstraint', box,'KernelFunction', kernel,'Polynomial', poly_order)
                    disp("Kernel is Polynomial")
                    %predict the label, score and store it in a table
                    [labelP, score] = predict(mdl, y)
                    t5=table(target_labels,labelP,score);
                    t5.Properties.VariableNames = {'TrueLabel','PredictedLabel','Score'};
                    Loss = resubLoss(mdl)
                    Loss_val(m) = [Loss];
            if mod(m,10)== 0
               %calculate the mean loss
               Loss1 = mean(Loss_val)  
               %plot confusion matrix and calculate accuracy
               CM_mdl = confusionmat(target_labels, labelP)
               Accuracy = 100*sum(diag(CM_mdl))./sum(CM_mdl(:));
               %define the columns of the matrix
               myrow ={kernel,box, poly_order, Loss1, Accuracy}
               mymatrix1 = [mymatrix1;myrow]      
             end      
                    end  
              end
         end
    end
end

%% creating the final matrix that collates all kernel, box constraint and polynomial order
%combinations and their respective accuracy
Final_matrix = [mymatrix; mymatrix1];

%%Automatically pick up the highest accuracy and its respective box
%%constraint, kernel function and polynomial order (if relevant) to fit
%%into the best model
highestAccuracy = max(Final_matrix{:,5})
best_model = Final_matrix(Final_matrix.Accuracy == highestAccuracy, :)
best_bc = best_model{:,2} ; 
best_kernel = best_model{:,1};
best_poly = best_model{:,3} ;

%% CREATING BEST MODEL FROM OPTIMISED PARAMETERS
net_SVM = fitcsvm(x,t,'BoxConstraint',best_bc,'Kernelfunction',best_kernel,'PolynomialOrder',best_poly);
%saving the best model
save('net_SVM.mat');

toc;


