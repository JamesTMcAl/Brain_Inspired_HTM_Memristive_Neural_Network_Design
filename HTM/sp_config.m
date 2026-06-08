classdef sp_config
    properties
        % Sparsity targets / overlap‐adaptation
        TARGET_ACTIVITY  = 0.30;
        MIN_DENSITY = 0.05;
        MAX_DENSITY = 0.60;
        ENTROPY_THRESHOLD_INIT = 0.50;
        SPARSITY_THRESHOLD_INIT = 35;

        OVERLAP_QUANTILE_INIT  = 0.90;
        OVERLAP_QUANTILE_UPDATE = 0.05;
        OVERLAP_QUANTILE_DECAY  = 0.95;
        OVERLAP_MIN_ACTIVE_FRAC = 0.2;  
        OVERLAP_MAX_ACTIVE_FRAC = 0.3;   
        Overlap_MIN_ACTV_PRE_TRSH = 0.1
        Overlap_MAX_ACTV_PRE_TRSH =  0.2
        % Synaptic defaults
        SYN_NOISE_LEVEL  = 0.10;   
        SYN_TARGET_RANGE  = [0.05,0.90];
        HASH_SUBSAMPLE  = 8;
        TRAIN_NOISE_BASE   = 0.02;    
        SYN_CONNECTED_INIT  = 0.25;
        NOISE_STD  = 0.05;   
        REFRACTORY_PERIOD   = 5;

        % Synaptic update scaling factors
        SYN_INC_SCALAR   = 1.5;    
        SYN_DEC_SCALAR   = 0.8;   
        SYN_INC_MIN_FACTOR    = 0.3;    
        SYN_INC_MAX_FACTOR   = 3.0;    
        SYN_DEC_MIN   = 0.001;  
        SYN_DEC_MAX    = 0.02;   
        STD_SCALING_FACTOR  = 0.003; 

        % k-NN classifier setting
        KNN_K  = 3;      
        
        % Hyperparameter tuning
        HYPER_SEED = 12345;  

        
        % PI‐controller gains
        KP_DENSITY  = 0.03;
        KI_DENSITY   = 0.005;
        KP_SYNAPTIC   = 0.10;
        MOMENTUM_GAIN  = 0.8;

        % kWTA / lateral inhibition
        KWTA_INIT_RADIUS   = 2;
        KWTA_RADIUS_SCALE  = 1.5;
        KWTA_RADIUS_HISTORY  = 10;
        KWTA_LOCAL_MAX   = 5;
        KWTA_LOCAL_SCALE  = 2;
        KWTA_LOCAL_MIN     = 2;
        INHIB_RADIUS_MAX   = 3;
        INHIB_RADIUS_MIN   = 1;
        KWTA_MIN_ACTIVE_FRAC  = 0.001;
        KWTA_FALLBACK_FACTOR  = 0.8;
        KWTA_TEMP_GROWTH_RATE     = 0.05;
        KWTA_TIME_CONST    = 2500;
        KWTA_BETA_BOOST = 0.03;
        KWTA_TARGET_DUTY = 0.05;
        % stochastic kWTA noise & pacing
        KWTA_NOISE_LEVEL  = 0.05;
        KWTA_STOCH_INTERVAL   = 500;
        KWTA_STOCH_BASE   = 0.1;
        KWTA_STOCH_MIN  = 0.01;
        KWTA_PI_PERIOD     = 32;
        BOOST_FACTOR    = 1.0;    

        % Debug / performance
        PCA_SAMPLE_FRAC = 0.10;    
        DEBUG       = false;
        DEBUG_ADJUST   = false;
        DEBUG_OVERLAP  = false;
        Debug_Overlap_Tracking = false;
        fallback_Debug = false;
        Debug_GPU = false;
        DEBUG_INTERVAL    = 100;
        EPOCH_START_FLAG     = false;
        OVERLAP_BATCH_SIZE  = 64;
        Debug_ENTROPY = false;
        USE_GPU = true;
        USE_PCA_INIT = true;
        SubsetTraining = 1000;
    end
    methods (Static)
        function obj = instance()
            persistent inst pathAdded        
            if isempty(pathAdded)
                try
                    addpath(fullfile( ...
                        fileparts(mfilename('fullpath')), ...
                        'utilsFolder'));
                catch ME
                    fprintf(2, 'sp_config.instance: could not add utilsFolder to path:\n%s\n', ME.message);
                end
                pathAdded = true;
            end
            if isempty(inst)
                inst = sp_config();
            end
            obj = inst;
        end
    end

    methods (Access = private)
        function obj = sp_config(), end
    end
end
