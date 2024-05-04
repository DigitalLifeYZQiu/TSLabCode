import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

import random
import numpy as np
from ckaCalculator.cka import CKACalculator

def ckaExample(args, model1_path,model2_path):
    
    exp = Exp_Long_Term_Forecast(args)
    device = exp._acquire_device()
    test_data, test_loader = exp._get_data(flag='test')

    model1 = exp._build_model().to(device)
    model1.load_state_dict(torch.load(model1_path))
    model2 = exp._build_model().to(device)
    model2.load_state_dict(torch.load(model2_path))

    layer_classes_model1 = tuple([layer.__class__ for name, layer in model1.named_modules()])
    layer_classes_model2 = tuple([layer.__class__ for name, layer in model2.named_modules()])
    layer_classes = tuple(set(layer_classes_model1) | set(layer_classes_model2))

    calculator = CKACalculator(
                pred_len=args.pred_len,label_len=args.label_len,batch_size=args.batch_size,
                device=device, model1=model1,model2=model2,dataloader=test_loader,hook_layer_types=layer_classes
            )
    cka_output = calculator.calculate_cka_matrix()
    print(f"CKA output size: {cka_output.size()}")
    for i, name in enumerate(calculator.module_names_X):
        print(f"Layer {i}: \t{name}")
    calculator.plot_cka_plotly(
        cka_matrix=cka_output,
        title="Model compared with itself",
        show_ticks_labels=True,
        short_tick_labels_splits=2,
        use_tight_layout=False,
        save_path="./CKAEXP",
        show_annotations=False,
        show_img=False,
    )

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')


    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    
    # cka analysis
    parser.add_argument('--use_cka',default=False, action="store_true", help="Centered Kernel Analysis for all network layers")

    args = parser.parse_args()


    model1_path = 'checkpoints/long_term_forecast_ETTh1_96_96_PatchTST_ETTh1_ftM_sl96_ll48_pl96_dm512_nh2_el1_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth'
    model2_path = 'checkpoints/long_term_forecast_ETTh1_96_96_PatchTST_ETTh1_ftM_sl96_ll48_pl96_dm512_nh2_el1_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth'
    ckaExample(args, model1_path,model2_path)
