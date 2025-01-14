import argparse, os
from pathlib import Path
import gin

@gin.configurable()
def build_model(args,model_fn=gin.REQUIRED):
    model = model_fn()
    return model


def aug_parse(parser: argparse.ArgumentParser):
    import yaml
    parser.add_argument('--no_resume',default=False,action='store_true',help="")
    parser.add_argument('--cfgs', nargs='+', default=[],
                        help='<Required> Config files *.gin.', required=False)
    parser.add_argument('--gin', nargs='+', 
                        help='Overrides config values. e.g. --gin "section.option=value"')
   
    args = parser.parse_args()
    
    if args.output_dir:
        output_dir=Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    gin.parse_config_files_and_bindings(args.cfgs,args.gin)
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(args.output_dir,'config.yml'), 'w') as f:
            yaml.dump(vars(args), f)
            
        open(output_dir/"config.gin",'w').write(gin.config_str(),)
    
    return args