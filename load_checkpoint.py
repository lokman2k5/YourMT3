# @title Load Checkpoint
model_name = 'YPTF.MoE+Multi (noPS)' # @param ["YMT3+", "YPTF+Single (noPS)", "YPTF+Multi (PS)", "YPTF.MoE+Multi (noPS)", "YPTF.MoE+Multi (PS)"]
precision = '16' # @param ["32", "bf16-mixed", "16"]
project = '2024'

if model_name == "YMT3+":
    checkpoint = "notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72@model.ckpt"
    args = [checkpoint, '-p', project, '-pr', precision]
elif model_name == "YPTF+Single (noPS)":
    checkpoint = "ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100@model.ckpt"
    args = [checkpoint, '-p', project, '-enc', 'perceiver-tf', '-ac', 'spec',
            '-hop', '300', '-atc', '1', '-pr', precision]
elif model_name == "YPTF+Multi (PS)":
    checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2@model.ckpt"
    args = [checkpoint, '-p', project, '-tk', 'mc13_full_plus_256',
            '-dec', 'multi-t5', '-nl', '26', '-enc', 'perceiver-tf',
            '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', precision]
elif model_name == "YPTF.MoE+Multi (noPS)":
    checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
    args = [checkpoint, '-p', project, '-tk', 'mc13_full_plus_256', '-dec', 'multi-t5',
            '-nl', '26', '-enc', 'perceiver-tf', '-sqr', '1', '-ff', 'moe',
            '-wf', '4', '-nmoe', '8', '-kmoe', '2', '-act', 'silu', '-epe', 'rope',
            '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', precision]
elif model_name == "YPTF.MoE+Multi (PS)":
    checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2@model.ckpt"
    args = [checkpoint, '-p', project, '-tk', 'mc13_full_plus_256', '-dec', 'multi-t5',
            '-nl', '26', '-enc', 'perceiver-tf', '-sqr', '1', '-ff', 'moe',
            '-wf', '4', '-nmoe', '8', '-kmoe', '2', '-act', 'silu', '-epe', 'rope',
            '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', precision]
else:
    raise ValueError(model_name)

model = load_model_checkpoint(args=args)
