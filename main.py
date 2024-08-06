import argparse
import json
from src.inference.result_maker import ResultMaker
from src.utils.visualize_results import VisualizeResult
from src.misc.read_config import load_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='analyse', choices=['analyse', 'visualize'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--config-path', type=str, default='configs/synthetic_config.yaml', help='The path to the configuration file to use (default: %(default)s)')
    parser.add_argument('--OSM-path', type=str, default=None, help='The path to the OSM data corresponding to the analysis. Needed for running TSAA.')
    parser.add_argument('--save-folder', type=str, default='experiment', help='The directory to store checkpoints and figures for the experiment (default: %(default)s)')
    parser.add_argument('--corrupt', action=argparse.BooleanOptionalAction, help='Whether to corrupt the data before analysis. If unspecified it is false')
    parser.add_argument('--rb', action=argparse.BooleanOptionalAction, help='Whether to have response bias in synthetic data. If unspecified it is false.')
    parser.add_argument('--M', type=int, default=20, help='The number of questions in the synthetitc questionnaire data.')
    parser.add_argument('--X-path', type=str, default=None, help='The path to a questionnaire dataset of dimensions M x N')
    parser.add_argument('--Z-path', type=str, default=None, help='The path to a ground-truth archetype matrix.')
    parser.add_argument('--A-path', type=str, default=None, help='The path to a ground-truth respondent weighting matrix.')
    parser.set_defaults(rb=False)
    parser.set_defaults(corrupt=False)
    args = parser.parse_args()
    
    cfg = load_config(path=args.config_path)
    cfg['data']['results']['checkpoint_dir'] = f"results/{args.save_folder}"
    cfg['visualization']['save_dir'] = f"figures/{args.save_folder}"

    if args.mode == 'analyse':
        cfg['data']['do_corrupt'] = args.corrupt
        cfg['data']['synthetic_data_params']['M'] = args.M
        cfg['data']['synthetic_data_params']['rb'] = args.rb
        cfg['data']['input_data_path'] = args.X_path
        cfg['data']['OSM_data_path'] = args.OSM_path
        cfg['data']['Z_path'] = args.Z_path
        cfg['data']['A_path'] = args.A_path
        
        RM = ResultMaker(cfg=cfg)
        if cfg['data']['use_synthetic_data']:
            RM.get_synthetic_results()
        
        else:
            print(cfg['data']['input_data_path'])
            print(cfg['data'])
            RM.get_real_results()
    
    elif args.mode == 'visualize':
        VR = VisualizeResult(cfg=cfg)
        
        TSOAA_result_path = cfg['visualization']['OSM_result_dir'] # none or path to dir
        VR.loss_archetype_plot(TSOAA_result_path=TSOAA_result_path)
        VR.NMI_archetype_plot(TSOAA_result_path=TSOAA_result_path)
        VR.NMI_stability_plot()
        VR.archetypal_answers_plot(method=cfg['visualization']['run_specific_analysis']['method'],
                                   K=cfg['visualization']['run_specific_analysis']['K'],
                                   rep=cfg['visualization']['run_specific_analysis']['rep'],
                                   type='lines') # points, lines
        
        VR.response_bias_plot(K = cfg['visualization']['run_specific_analysis']['K'], 
                              rep=cfg['visualization']['run_specific_analysis']['rep'])
        if args.corrupt: # plot corruption error if relevant
            VR.denoising_plot()
        
        
        print("Successfully created visualizations. The plots have been saved to {write_dir}".format(write_dir=cfg['visualization']['save_dir']))
        
