clear

%load the matrices from eeg.mat
eeg=load ('eeg.mat');

x_te=eeg.x_te;

x_train=eeg.x_train;

y_te=eeg.y_te;

y_train=eeg.y_train;

%Loads the class labels of the training data for classification of the test
%data, Binary_Hash_Training vector (:,n) is classified by
%Training_classes scalar (1,n)
Training_Classes=y_train';

%Training Data STFT
%Creates 3 matrices (1 for each eeg channel) to store STFTs of each trial
for n=1:112

    %Creates 3 matrices (1 for each eeg channel) to store STFTs of each trial
    X_Tr_Spec_ch1(:,:,n)=spectrogram(x_train(:,1,n),blackman(64),16,64); %STFT of ch1

    X_Tr_Spec_ch2(:,:,n)=spectrogram(x_train(:,2,n),blackman(64),16,64); %STFT of ch2

    X_Tr_Spec_ch3(:,:,n)=spectrogram(x_train(:,3,n),blackman(64),16,64); %STFT of ch3
    
    %extract the mu wave signal from all of the channels and all of the trials
    Ch1_Vector=cat(2,X_Tr_Spec_ch1(3,:,n),X_Tr_Spec_ch1(4,:,n),X_Tr_Spec_ch1(5,:,n),X_Tr_Spec_ch1(6,:,n),X_Tr_Spec_ch1(7,:,n))';
    
    Ch2_Vector=cat(2,X_Tr_Spec_ch2(3,:,n),X_Tr_Spec_ch2(4,:,n),X_Tr_Spec_ch2(5,:,n),X_Tr_Spec_ch2(6,:,n),X_Tr_Spec_ch2(7,:,n))';
    
    Ch3_Vector=cat(2,X_Tr_Spec_ch3(3,:,n),X_Tr_Spec_ch3(4,:,n),X_Tr_Spec_ch3(5,:,n),X_Tr_Spec_ch3(6,:,n),X_Tr_Spec_ch3(7,:,n))';
    
    %loads the input sample vectors into a matrix of all 112 trials
    Sample_Mat(:,n)=cat(1,Ch1_Vector,Ch2_Vector,Ch3_Vector);
    
end

%Test Data STFT
for n=1:28

    %Creates 3 matrices (1 for each eeg channel) to store STFTs of each trial
    X_Te_Spec_ch1(:,:,n)=spectrogram(x_te(:,1,n),blackman(64),16,64); %STFT of ch1

    X_Te_Spec_ch2(:,:,n)=spectrogram(x_te(:,2,n),blackman(64),16,64); %STFT of ch2

    X_Te_Spec_ch3(:,:,n)=spectrogram(x_te(:,3,n),blackman(64),16,64); %STFT of ch3
    
    %extract the mu wave signal from all of the channels and all of the trials
    Te_Ch1_Vector=cat(2,X_Te_Spec_ch1(3,:,n),X_Te_Spec_ch1(4,:,n),X_Te_Spec_ch1(5,:,n),X_Te_Spec_ch1(6,:,n),X_Te_Spec_ch1(7,:,n))';
    
    Te_Ch2_Vector=cat(2,X_Te_Spec_ch2(3,:,n),X_Te_Spec_ch2(4,:,n),X_Te_Spec_ch2(5,:,n),X_Te_Spec_ch2(6,:,n),X_Te_Spec_ch2(7,:,n))';
    
    Te_Ch3_Vector=cat(2,X_Te_Spec_ch3(3,:,n),X_Te_Spec_ch3(4,:,n),X_Te_Spec_ch3(5,:,n),X_Te_Spec_ch3(6,:,n),X_Te_Spec_ch3(7,:,n))';
    
    %loads the input sample vectors into a matrix of all 112 trials
    Test_Sample_Mat(:,n)=cat(1,Te_Ch1_Vector,Te_Ch2_Vector,Te_Ch3_Vector);
    
end

%Perform PCA algorithm
%----------------------------------------------------------------------------

%Eigenvalue decomposition on Training Data
%Calculate and subtract the mean
Sample_mean=mean(Sample_Mat,2);
Sample_zm=Sample_Mat-Sample_mean;

%Find the covariance
Sample_cov=Sample_zm*Sample_zm'; %Cov(X)=XX^T

%Power iteration to find the eigenvectors

EIG=50; %number of eigenvectors/eigenvalues to compute

Eigvec_Mat = rand(225,EIG); %this is the random seed for the start of the power iteration

X_cov_it=Sample_cov;

for e=1:EIG;
    for x=1:100 %iterate 100 times
        Pwr_it_num = X_cov_it*Eigvec_Mat(:,e);
        Pwr_it_dem = norm(Pwr_it_num);
        Eigvec_Mat(:,e)=(Pwr_it_num/Pwr_it_dem);
        %disp(Eigvec_Mat)
    end

    Eig_Val(1,e) = Pwr_it_dem; %Load Eigenvalue into Eigenvalue vector

    %compute new matrix to find next biggest eigenvector and value
    EET=(Eigvec_Mat(:,e)*Eigvec_Mat(:,e)'); %multiply largest eigenvector by its transpose
    X_cov_it=X_cov_it-Eig_Val(1,e)*EET;%multiply that by eigen value and subtract from original matrix
end

%Plot eigenvalues to determine how many PCs to keep
plot(Eig_Val)
title('Eigenvalues for EEG Sample')
text(10,500,'\downarrow Effect of Eigenvalues minimal beyond here')


%Peform Eigenvalue decomposition on Test Data
%--------------------------------------------------
%Calculate and subtract the mean
Test_Sample_mean=mean(Test_Sample_Mat,2);
Test_Sample_zm=Test_Sample_Mat-Test_Sample_mean;

%Find the covariance
Test_Sample_cov=Test_Sample_zm*Test_Sample_zm'; %Cov(X)=XX^T

%Power iteration to find the eigenvectors

Test_EIG=28; %number of eigenvectors/eigenvalues to compute

Test_Eigvec_Mat = rand(225,Test_EIG); %this is the random seed for the start of the power iteration

Test_X_cov_it=Test_Sample_cov;

for e=1:Test_EIG;
    for x=1:100 %iterate 100 times
        Pwr_it_num = Test_X_cov_it*Test_Eigvec_Mat(:,e);
        Pwr_it_dem = norm(Pwr_it_num);
        Test_Eigvec_Mat(:,e)=(Pwr_it_num/Pwr_it_dem);
        
    end

    Test_Eig_Val(1,e) = Pwr_it_dem; %Load Eigenvalue into Eigenvalue vector

    %compute new matrix to find next biggest eigenvector and value
    Test_EET=(Test_Eigvec_Mat(:,e)*Test_Eigvec_Mat(:,e)'); %multiply largest eigenvector by its transpose
    Test_X_cov_it=Test_X_cov_it-Test_Eig_Val(1,e)*Test_EET;%multiply that by eigen value and subtract from original matrix
end

%Vary the Number of PCs to test effect of different numbers

Error_X=1;
Error_Y=1;
Error_Z=1;

for PC=2:4:22 %vary the # of PCs from 2 to 22 in steps of 4
 
    %Perform PCA on Training Data
  %--------------------------------------------  
    Z_PCA=Eigvec_Mat(:,1:PC);

    %Setup the Diagonal Matrix of Eigenvalues
    Eig_Val_Whitening=1./sqrt(Eig_Val(1,1:PC));
    Delta_Whitening=zeros(PC,PC);

    %Diagonalize the Whitening Eigenvalues

    for it=1:PC
    Delta_Whitening(it,it)=Eig_Val_Whitening(1,it);

    end

    %Whiten and perform PCA on Sample Matrix
    Z=Delta_Whitening*Z_PCA'*Sample_Mat;
%-----------------------------------------------

    %Whiten and reduce dimensions of test data
    %--------------------------------------------------------
    %!!!!Use same number of eigenvectors (PC) as training data!!!!

    Test_Z_PCA=Test_Eigvec_Mat(:,1:PC);

    %Setup the Diagonal Matrix of Eigenvalues
    Test_Eig_Val_Whitening=1./sqrt(Test_Eig_Val(1,1:PC));
    Test_Delta_Whitening=zeros(PC,PC);

    %Diagonalize the Whitening Eigenvalues
    for it=1:PC
    Test_Delta_Whitening(it,it)=Test_Eig_Val_Whitening(1,it);

    end

    %Whiten and perform PCA on Sample Matrix
    Test_Z=Test_Delta_Whitening*Test_Z_PCA'*Test_Sample_Mat;

    %----------------------------------------------------------------------
    %Develop Random Projection Matrix of size (L X M)
    M=PC; % # of PC eignvectors chosen earlier
   for L=5:5:25 % rand dimension L

        A=rand(L,M);

        %Normalize the row vectors
        for it=1:L
            Norm_It=norm(A(it,:));
            A(it,:)=A(it,:)/Norm_It;
        end

        %Apply the projection to the dimensionally reduced Sample Matrix
        Y=sign(A*Z);

        %Convert Y into binary hash codes 0 or 1
        Binary_Hash_Training=Y>0;
        
        Binary_Hash_Training=double(Binary_Hash_Training); %Recast as double to remove repeated warnings


        %------------------------------------------------------------------------

        %Apply the same A projection from training to the dimensionally reduced Test Matrix
        Test_Y=sign(A*Test_Z);

        %Convert Test_Y into binary hash codes 0 or 1
        Binary_Hash_Testing=Test_Y>0;
        
        Binary_Hash_Testing=double(Binary_Hash_Testing); %Recast as double to remove repeated warnings

        %Perform kNN classification
        for K=5:5:15; %alter K for different performance results

            %creates a matrix to capture the 28 test samples hamming distance
            %to the 112 training samples
            for te_sample=1:28
                for tr_sample=1:112

                    %finds the hamming distance between the test sample and the training samples as a decimal ratio
                    Dist_te(1,tr_sample)=pdist2(Binary_Hash_Training(:,tr_sample)',Binary_Hash_Testing(:,te_sample)','hamming');

                end

                %sorts the Hamming distances from smallest to largest and captures the
                %index of each in Dist_Sort
                [Dist_Value,Dist_Sort]=sort(Dist_te);

                %keeps the K nearest neighbors in kNN
                kNN(1:K)=Dist_Sort(1:K);

                %maps the kNN to classes from y_train (Training_Classes)
                for it=1:K

                    kNN_Classes(it)=Training_Classes(kNN(it));

                    Bins=histcounts(kNN_Classes,2); %counts how many are from each class out of the nearest neighbors
                end

                %if there is more of Bin(1) which counts the # of 1's class than the
                %kNN result is one, else if more of Bin(2) which counts the # of 2's
                %class than the kNN result is two
                if Bins(1)>Bins(2)

                    Test_Results(te_sample)=1;
                else
                    Test_Results(te_sample)=2;
                end
            end

            %Test results of classification compared to ground truth data y_te
            Comparison_Y=y_te-Test_Results';

            Error_Rate=histcounts(Comparison_Y,3); %determines how many zeros there are ZERO=correct classification

            Class_Percentage=Error_Rate(2)/28; %calculates the percentage of correction classifications as a decimal

            Class_Error_Effects(Error_X,Error_Y,Error_Z)=Class_Percentage;
            
            Error_Z=K/5;
            
        end
   
        Error_Y=L/5;
        
   end
   
   Error_X=Error_X+1;
   
end

%plot the different values K=5
figure()
labels = [2 6 10 14 16 22];
bar(100*Class_Error_Effects(:,:,1))
set(gca, 'XTick', 1:length(labels)); % Change x-axis ticks
set(gca, 'XTickLabel', labels); % Change x-axis ticks labels.
ylabel('Percent Correct Classification')
xlabel('Number of Principal Components')
legend('L=5','L=10','L=15','L=20','L=25')
title('Correct Classification Percentage (K=5)')

%plot the different values K=10
figure()
labels = [2 6 10 14 16 22];
bar(100*Class_Error_Effects(:,:,2))
set(gca, 'XTick', 1:length(labels)); % Change x-axis ticks
set(gca, 'XTickLabel', labels); % Change x-axis ticks labels.
ylabel('Percent Correct Classification')
xlabel('Number of Principal Components')
legend('L=5','L=10','L=15','L=20','L=25')
title('Correct Classification Percentage (K=10)')

%plot the different values K=15
figure()
labels = [2 6 10 14 16 22];
bar(100*Class_Error_Effects(:,:,3))
set(gca, 'XTick', 1:length(labels)); % Change x-axis ticks
set(gca, 'XTickLabel', labels); % Change x-axis ticks labels.
ylabel('Percent Correct Classification')
xlabel('Number of Principal Components')
legend('L=5','L=10','L=15','L=20','L=25')
title('Correct Classification Percentage (K=15)')

