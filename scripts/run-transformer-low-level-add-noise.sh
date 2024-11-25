pointcloud_num=4500

cd 3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy

# source_dir="/scratch/chialiang/dp3_demo_combine_2_new"
source_dir="/scratch/yufeiw2/dp3_demo_combine_2_new"
source_dir2="/scratch/yufeiw2/dp3_demo_combined_2_step"

observation_mode="act3d_goal_mlp"
# observation_mode='act3d_goal_mlp_displacement_gripper_to_object'
encoding_mode="keep_position_feature_in_attention_feature"

horizon=8
n_obs_steps=2 # 2 or 4

##########
training_epoches=100
train_ratio=0.9 # for generalization
num_load_episodes=1000    # for generalization
pc_channel=3 # we should modify this
batch_size=30 #######
# batch_size=112 #######
encoder_type=act3d
use_mlp=1
use_lightweight_unet=0
in_channels=3 ####
self_attention=false
final_attention=false
normalize_action=true
augmentation_rot=false
augmentation_pcd=true
use_absolute_waypoint=false
dense_pcd_for_goal=false
##########
use_attn_for_point_features=false
pointcloud_backbone='mlp'
##########
is_pickle=true
##########
use_pretrained_high_level_policy_as_low_level_input=false
##########

time_stamp=$(date +%m%d%H%M)
exp_name="1111-600-combined-low-level-transformer-diffusion-w-learned-high-level-and-goal-noise"

action_dim=10
agent_pos_dim=10

# saved data paths
save_data_name_0="0622-act3d-obj-45448-remove-reaching-collision-resize-2-full-per-step-gripper-goal-displacement-to-closest-obj-point"
save_data_name_1="0624-act3d-obj-46462-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action"
save_data_name_2="0626-act3d-obj-41510-per-step-combine-2-action-gripper-goal-displacement-to-closest-obj-point-filtered-zero-closing-action"
save_data_name_3="0628-act3d-obj-46732-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1"
save_data_name_4="0628-act3d-obj-46801-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1"
save_data_name_5="0628-act3d-obj-46874-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1"
save_data_name_6="0628-act3d-obj-46922-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1"
save_data_name_7="0628-act3d-obj-46966-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1"
save_data_name_8="0628-act3d-obj-47570-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1"
save_data_name_9="0628-act3d-obj-47578-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1"
save_data_name_10="0628-act3d-obj-48700-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1"    
save_data_name_11="0705-obj-45526"
save_data_name_12="0705-obj-45661"
save_data_name_13="0705-obj-45694"
save_data_name_14="0705-obj-45780"
save_data_name_15="0705-obj-45910"
save_data_name_16="0705-obj-45961"
save_data_name_17="0705-obj-46408"
save_data_name_18="0705-obj-46417"
save_data_name_19="0705-obj-46440"
save_data_name_20="0705-obj-46490"
save_data_name_21="0705-obj-46762"
save_data_name_22="0705-obj-46825"
save_data_name_23="0705-obj-46893"
save_data_name_24="0705-obj-47235"
save_data_name_25="0705-obj-47281"
save_data_name_26="0705-obj-47315"
save_data_name_27="0705-obj-47529"
save_data_name_28="0705-obj-47669"
save_data_name_29="0705-obj-47944"
save_data_name_30="0705-obj-48063"
save_data_name_31="0705-obj-48177"
save_data_name_32="0705-obj-48356"
save_data_name_33="0705-obj-48623"
save_data_name_34="0705-obj-48876"
save_data_name_35="0705-obj-49025"
save_data_name_36="0705-obj-49062"
save_data_name_37="0705-obj-49132"
save_data_name_38="0705-obj-49133"
save_data_name_39="0712-obj-40417"
save_data_name_40="0712-obj-41085"
save_data_name_41="0712-obj-41452"
save_data_name_42="0712-obj-45162"
save_data_name_43="0712-obj-45176"
save_data_name_44="0712-obj-45194"
save_data_name_45="0712-obj-45203"
save_data_name_46="0712-obj-45248"
save_data_name_47="0712-obj-45271"
save_data_name_48="0712-obj-45290"
save_data_name_49="0712-obj-45305"
save_data_name_50="0725-obj-45427"


save_data_name_50=0725-obj-45427
save_data_name_51=0725-obj-45620
save_data_name_52=0725-obj-45623
save_data_name_53=0725-obj-45636
save_data_name_54=0725-obj-45689
save_data_name_55=0725-obj-45696
save_data_name_56=0725-obj-45749
save_data_name_57=0725-obj-45759
save_data_name_58=0725-obj-45936
save_data_name_59=0725-obj-45984
save_data_name_60=0725-obj-46130
save_data_name_61=0725-obj-46197
save_data_name_62=0725-obj-46481
save_data_name_63=0725-obj-46544
save_data_name_64=0725-obj-47178
save_data_name_65=0725-obj-47182
save_data_name_66=0725-obj-47227
save_data_name_67=0725-obj-47577
save_data_name_68=0725-obj-47648
save_data_name_69=0725-obj-47747
save_data_name_70=0725-obj-47808
save_data_name_71=0725-obj-47976
save_data_name_72=0725-obj-48010
save_data_name_73=0725-obj-48258
save_data_name_74=0725-obj-48379
save_data_name_75=0725-obj-48797
save_data_name_76=0725-obj-48855
save_data_name_77=0725-obj-48859
save_data_name_78=0725-obj-49188
save_data_name_79=0730-obj-35059
save_data_name_80=0730-obj-41004
save_data_name_81=0730-obj-41083
save_data_name_82=0730-obj-44781
save_data_name_83=0730-obj-44826
save_data_name_84=0730-obj-44853
save_data_name_85=0730-obj-45092
save_data_name_86=0730-obj-45130
save_data_name_87=0730-obj-45135
save_data_name_88=0730-obj-45146
save_data_name_89=0730-obj-45164
save_data_name_90=0730-obj-45168
save_data_name_91=0730-obj-45173
save_data_name_92=0730-obj-45212
save_data_name_93=0730-obj-45213
save_data_name_94=0730-obj-45372
save_data_name_95=0730-obj-45374
save_data_name_96=0730-obj-45387
save_data_name_97=0730-obj-45415
save_data_name_98=0730-obj-45419
save_data_name_99=0730-obj-45423
save_data_name_100=0730-obj-45503
save_data_name_101=0730-obj-45505
save_data_name_102=0730-obj-45524
save_data_name_103=0730-obj-45573
save_data_name_104=0730-obj-45575
save_data_name_105=0730-obj-45606
save_data_name_106=0730-obj-45612
save_data_name_107=0730-obj-45621
save_data_name_108=0730-obj-45622
save_data_name_109=0730-obj-45632
save_data_name_110=0730-obj-45638
save_data_name_111=0730-obj-45645
save_data_name_112=0730-obj-45662
save_data_name_113=0730-obj-45671
save_data_name_114=0730-obj-45676
save_data_name_115=0730-obj-45677
save_data_name_116=0730-obj-45687
save_data_name_117=0730-obj-45699
save_data_name_118=0730-obj-45710
save_data_name_119=0730-obj-45746
save_data_name_120=0730-obj-45756
save_data_name_121=0730-obj-45783
save_data_name_122=0730-obj-45784
save_data_name_123=0730-obj-45790
save_data_name_124=0730-obj-45801
save_data_name_125=0730-obj-45822
save_data_name_126=0730-obj-45853
save_data_name_127=0730-obj-45855
save_data_name_128=0730-obj-45915
save_data_name_129=0730-obj-45948
save_data_name_130=0730-obj-45949
save_data_name_131=0730-obj-45963
save_data_name_132=0730-obj-45964
save_data_name_133=0730-obj-46019
save_data_name_134=0730-obj-46029
save_data_name_135=0730-obj-46033
save_data_name_136=0730-obj-46037
save_data_name_137=0730-obj-46044
save_data_name_138=0730-obj-46045
save_data_name_139=0730-obj-46060
save_data_name_140=0730-obj-46084
save_data_name_141=0730-obj-46108
save_data_name_142=0730-obj-46117
save_data_name_143=0730-obj-46120
save_data_name_144=0730-obj-46123
save_data_name_145=0730-obj-46145
save_data_name_146=0730-obj-46179
save_data_name_147=0730-obj-46180
save_data_name_148=0730-obj-46199
save_data_name_149=0730-obj-46380
save_data_name_150=0730-obj-46427
save_data_name_151=0730-obj-46430
save_data_name_152=0730-obj-46439
save_data_name_153=0730-obj-46537
save_data_name_154=0730-obj-46549
save_data_name_155=0730-obj-46556
save_data_name_156=0730-obj-46598
save_data_name_157=0730-obj-46616
save_data_name_158=0730-obj-46699
save_data_name_159=0730-obj-46700
save_data_name_160=0730-obj-46741
save_data_name_161=0730-obj-46744
save_data_name_162=0730-obj-46847
save_data_name_163=0730-obj-46856
save_data_name_164=0730-obj-46859
save_data_name_165=0730-obj-46889
save_data_name_166=0730-obj-46906
save_data_name_167=0730-obj-46944
save_data_name_168=0730-obj-46955
save_data_name_169=0730-obj-46981
save_data_name_170=0730-obj-47024
save_data_name_171=0730-obj-47089
save_data_name_172=0730-obj-47183
save_data_name_173=0730-obj-47207
save_data_name_174=0730-obj-47233
save_data_name_175=0730-obj-47252
save_data_name_176=0730-obj-47278
save_data_name_177=0730-obj-47290
save_data_name_178=0730-obj-47296
save_data_name_179=0730-obj-47438
save_data_name_180=0730-obj-47514
save_data_name_181=0730-obj-47595
save_data_name_182=0730-obj-47601
save_data_name_183=0730-obj-47632
save_data_name_184=0730-obj-47701
save_data_name_185=0730-obj-47729
save_data_name_186=0730-obj-47853
save_data_name_187=0730-obj-47926
save_data_name_188=0730-obj-48413
save_data_name_189=0730-obj-48452
save_data_name_190=0730-obj-48467
save_data_name_191=0730-obj-48490
save_data_name_192=0730-obj-48513
save_data_name_193=0730-obj-48517
save_data_name_194=0730-obj-48721
save_data_name_195=0730-obj-48746
save_data_name_196=0730-obj-48878

save_data_name_197='0725-obj-41003'
save_data_name_198='0725-obj-45001'
save_data_name_199='0725-obj-45235'
save_data_name_200='0725-obj-45238'
save_data_name_201='0725-obj-45244'
save_data_name_202='0725-obj-45249'
save_data_name_203='0705-obj-45523'
save_data_name_204='0705-obj-46014'
save_data_name_205='0705-obj-46166'
save_data_name_206='0705-obj-46653'
save_data_name_207='0705-obj-47711'
save_data_name_208='0705-obj-48263'
save_data_name_209='0730-obj-45007'
save_data_name_210='0730-obj-45087'
save_data_name_211='0730-obj-45159'
save_data_name_212='0730-obj-45166'
save_data_name_213='0730-obj-45189'
save_data_name_214='0730-obj-45247'
save_data_name_215='0730-obj-45261'
save_data_name_216='0730-obj-45267'
save_data_name_217='0730-obj-45354'
save_data_name_218='0725-obj-45413'
save_data_name_219='0725-obj-45420'
save_data_name_220='0725-obj-45594'
save_data_name_221='0725-obj-45670'
save_data_name_222='0725-obj-45916'
save_data_name_223='0725-obj-45950'
save_data_name_224='0725-obj-46092'
save_data_name_225='0725-obj-46134'
save_data_name_226='0730-obj-46230'
save_data_name_227='0730-obj-46277'
save_data_name_228='0725-obj-46334'
save_data_name_229='0725-obj-46443'
save_data_name_230='0730-obj-46466'
save_data_name_231='0725-obj-46480'
save_data_name_232='0725-obj-46641'
save_data_name_233='0730-obj-47088'
save_data_name_234='0730-obj-47185'
save_data_name_235='0725-obj-47254'
save_data_name_236='0730-obj-47419'
save_data_name_237='0730-obj-47613'
save_data_name_238='0725-obj-47742'
save_data_name_239='0730-obj-48018'
save_data_name_240='0730-obj-48023'
save_data_name_241='0730-obj-48051'
save_data_name_242='0730-obj-48271'
save_data_name_243='0730-obj-48491'
save_data_name_244='0730-obj-48519'
save_data_name_245='0730-obj-48740'
save_data_name_246='0730-obj-49140'
save_data_name_247='0822-obj-10036'
save_data_name_248='0822-obj-10143'
save_data_name_249='0822-obj-10144'
save_data_name_250='0822-obj-10655'
save_data_name_251='0822-obj-10797'
save_data_name_252='0822-obj-10867'
save_data_name_253='0822-obj-10944'
save_data_name_254='0822-obj-11211'
save_data_name_255='0822-obj-11661'
save_data_name_256='0822-obj-11700'
save_data_name_257='0822-obj-12042'
save_data_name_258='0822-obj-12043'
save_data_name_259='0822-obj-12259'
save_data_name_260='0822-obj-12480'
save_data_name_261='0822-obj-12531'
save_data_name_262='0822-obj-12536'
save_data_name_263='0822-obj-12543'
save_data_name_264='0822-obj-12552'
save_data_name_265='0822-obj-12553'
save_data_name_266='0822-obj-12559'
save_data_name_267='0822-obj-12561'
save_data_name_268='0822-obj-12562'
save_data_name_269='0822-obj-12563'
save_data_name_270='0822-obj-12579'
save_data_name_271='0822-obj-12583'
save_data_name_272='0822-obj-12587'
save_data_name_273='0822-obj-12590'
save_data_name_274='0822-obj-12592'
save_data_name_275='0822-obj-12594'
save_data_name_276='0822-obj-12596'
save_data_name_277='0822-obj-12605'
save_data_name_278='0822-obj-12606'
save_data_name_279='0822-obj-12614'
save_data_name_280='0822-obj-12617'
save_data_name_281='0822-obj-7119'
save_data_name_282='0822-obj-7167'
save_data_name_283='0822-obj-7187'
save_data_name_284='0822-obj-7220'
save_data_name_285='0822-obj-7263'
save_data_name_286='0822-obj-7290'

save_data_name_287="0815-obj-35059" 
save_data_name_288="0815-obj-41004" 
save_data_name_289="0815-obj-44781" 
save_data_name_290="0815-obj-44826" 
save_data_name_291="0815-obj-45087" 
save_data_name_292="0815-obj-45092" 
save_data_name_293="0815-obj-45130" 
save_data_name_294="0815-obj-45135" 
save_data_name_295="0815-obj-45164" 
save_data_name_296="0815-obj-45168" 
save_data_name_297="0815-obj-45173" 
save_data_name_298="0815-obj-45189" 
save_data_name_299="0815-obj-45212" 
save_data_name_300="0815-obj-45213" 
save_data_name_301="0815-obj-45247" 
save_data_name_302="0815-obj-45261" 
save_data_name_303="0815-obj-45267" 
save_data_name_304="0815-obj-45354" 
save_data_name_305="0815-obj-45387" 
save_data_name_306="0815-obj-45413" 
save_data_name_307="0815-obj-45419" 
save_data_name_308="0815-obj-45420" 
save_data_name_309="0815-obj-45423" 
save_data_name_310="0815-obj-45503" 
save_data_name_311="0815-obj-45505" 
save_data_name_312="0815-obj-45524" 
save_data_name_313="0815-obj-45573" 
save_data_name_314="0815-obj-45575" 
save_data_name_315="0815-obj-45594" 
save_data_name_316="0815-obj-45606" 
save_data_name_317="0815-obj-45620" 
save_data_name_318="0815-obj-45623" 
save_data_name_319="0815-obj-45638" 
save_data_name_320="0815-obj-45645" 
save_data_name_321="0815-obj-45662" 
save_data_name_322="0815-obj-45677" 
save_data_name_323="0815-obj-45699" 
save_data_name_324="0815-obj-45746" 
save_data_name_325="0815-obj-45783" 
save_data_name_326="0815-obj-45790" 
save_data_name_327="0815-obj-45822" 
save_data_name_328="0815-obj-45853" 
save_data_name_329="0815-obj-45855" 
save_data_name_330="0815-obj-45936" 
save_data_name_331="0815-obj-45937" 
save_data_name_332="0815-obj-45948" 
save_data_name_333="0815-obj-45963" 
save_data_name_334="0815-obj-45964" 
save_data_name_335="0815-obj-46029" 
save_data_name_336="0815-obj-46037" 
save_data_name_337="0815-obj-46060" 
save_data_name_338="0815-obj-46092" 
save_data_name_339="0815-obj-46117" 
save_data_name_340="0815-obj-46120" 
save_data_name_341="0815-obj-46132" 
save_data_name_342="0815-obj-46179" 
save_data_name_343="0815-obj-46197" 
save_data_name_344="0815-obj-46430" 
save_data_name_345="0815-obj-46456" 
save_data_name_346="0815-obj-46537" 
save_data_name_347="0815-obj-46556" 
save_data_name_348="0815-obj-46616" 
save_data_name_349="0815-obj-46699" 
save_data_name_350="0815-obj-46744" 
save_data_name_351="0815-obj-46847" 
save_data_name_352="0815-obj-46889" 
save_data_name_353="0815-obj-46906" 
save_data_name_354="0815-obj-47024" 
save_data_name_355="0815-obj-47178" 
save_data_name_356="0815-obj-47207" 
save_data_name_357="0815-obj-47227" 
save_data_name_358="0815-obj-47252" 
save_data_name_359="0815-obj-47296" 
save_data_name_360="0815-obj-47419" 
save_data_name_361="0815-obj-47438" 
save_data_name_362="0815-obj-47577" 
save_data_name_363="0815-obj-47595" 
save_data_name_364="0815-obj-47601" 
save_data_name_365="0815-obj-47613" 
save_data_name_366="0815-obj-47632" 
save_data_name_367="0815-obj-47729" 
save_data_name_368="0815-obj-47742" 
save_data_name_369="0815-obj-47747" 
save_data_name_370="0815-obj-47808" 
save_data_name_371="0815-obj-47853" 
save_data_name_372="0815-obj-47976" 
save_data_name_373="0815-obj-48010" 
save_data_name_374="0815-obj-48023" 
save_data_name_375="0815-obj-48258" 
save_data_name_376="0815-obj-48271" 
save_data_name_377="0815-obj-48379" 
save_data_name_378="0815-obj-48413" 
save_data_name_379="0815-obj-48452" 
save_data_name_380="0815-obj-48467" 
save_data_name_381="0815-obj-48490" 
save_data_name_382="0815-obj-48513" 
save_data_name_383="0815-obj-48517" 
save_data_name_384="0815-obj-48721" 
save_data_name_385="0815-obj-48740" 
save_data_name_386="0815-obj-48746" 
save_data_name_387="0815-obj-48797" 
save_data_name_388="0815-obj-48855" 
save_data_name_389="0815-obj-48859" 
save_data_name_390="0815-obj-48878" 
save_data_name_391="0815-obj-49188" 
save_data_name_392="0826-obj-44781" 
save_data_name_393="0826-obj-45092" 
save_data_name_394="0826-obj-45135" 
save_data_name_395="0826-obj-45159" 
save_data_name_396="0826-obj-45213" 
save_data_name_397="0826-obj-45247" 
save_data_name_398="0826-obj-45261" 
save_data_name_399="0826-obj-45354" 
save_data_name_400="0826-obj-45372" 
save_data_name_401="0826-obj-45387" 
save_data_name_402="0826-obj-45423" 
save_data_name_403="0826-obj-45575" 
save_data_name_404="0826-obj-45606" 
save_data_name_405="0826-obj-45621" 
save_data_name_406="0826-obj-45638" 
save_data_name_407="0826-obj-45645" 
save_data_name_408="0826-obj-45662" 
save_data_name_409="0826-obj-45676" 
save_data_name_410="0826-obj-45677" 
save_data_name_411="0826-obj-45746" 
save_data_name_412="0826-obj-45790" 
save_data_name_413="0826-obj-45853" 
save_data_name_414="0826-obj-45855" 
save_data_name_415="0826-obj-45915" 
save_data_name_416="0826-obj-45963" 
save_data_name_417="0826-obj-45964" 
save_data_name_418="0826-obj-46002" 
save_data_name_419="0826-obj-46029" 
save_data_name_420="0826-obj-46033" 
save_data_name_421="0826-obj-46037" 
save_data_name_422="0826-obj-46044" 
save_data_name_423="0826-obj-46060" 
save_data_name_424="0826-obj-46117" 
save_data_name_425="0826-obj-46120" 
save_data_name_426="0826-obj-46179" 
save_data_name_427="0826-obj-46430" 
save_data_name_428="0826-obj-46439" 
save_data_name_429="0826-obj-46537" 
save_data_name_430="0826-obj-46616" 
save_data_name_431="0826-obj-46699" 
save_data_name_432="0826-obj-46847" 
save_data_name_433="0826-obj-46856" 
save_data_name_434="0826-obj-46859" 
save_data_name_435="0826-obj-46889" 
save_data_name_436="0826-obj-46955" 
save_data_name_437="0826-obj-47024" 
save_data_name_438="0826-obj-47088" 
save_data_name_439="0826-obj-47089" 
save_data_name_440="0826-obj-47207" 
save_data_name_441="0826-obj-47233" 
save_data_name_442="0826-obj-47252" 
save_data_name_443="0826-obj-47296" 
save_data_name_444="0826-obj-47388" 
save_data_name_445="0826-obj-47419" 
save_data_name_446="0826-obj-47595" 
save_data_name_447="0826-obj-47601" 
save_data_name_448="0826-obj-47613" 
save_data_name_449="0826-obj-47632" 
save_data_name_450="0826-obj-47701" 
save_data_name_451="0826-obj-47729" 
save_data_name_452="0826-obj-47853" 
save_data_name_453="0826-obj-48023" 
save_data_name_454="0826-obj-48271" 
save_data_name_455="0826-obj-48413" 
save_data_name_456="0826-obj-48467" 
save_data_name_457="0826-obj-48490" 
save_data_name_458="0826-obj-48491" 
save_data_name_459="0826-obj-48513" 
save_data_name_460="0826-obj-48517" 
save_data_name_461="0826-obj-48721" 
save_data_name_462="0826-obj-48740"

# with gripper opening at the first frame
save_data_name_463="1026-obj-10143" 
save_data_name_464="1026-obj-10144" 
save_data_name_465="1026-obj-10489" 
save_data_name_466="1026-obj-10655" 
save_data_name_467="1026-obj-10944" 
save_data_name_468="1026-obj-11178" 
save_data_name_469="1026-obj-11211" 
save_data_name_470="1026-obj-11304" 
save_data_name_471="1026-obj-11550" 
save_data_name_472="1026-obj-11661" 
save_data_name_473="1026-obj-12043" 
save_data_name_474="1026-obj-12054" 
save_data_name_475="1026-obj-12252" 
save_data_name_476="1026-obj-12259" 
save_data_name_477="1026-obj-12530" 
save_data_name_478="1026-obj-12531" 
save_data_name_479="1026-obj-12536" 
save_data_name_480="1026-obj-12540" 
save_data_name_481="1026-obj-12552" 
save_data_name_482="1026-obj-12553" 
save_data_name_483="1026-obj-12559" 
save_data_name_484="1026-obj-12561" 
save_data_name_485="1026-obj-12579" 
save_data_name_486="1026-obj-12580" 
save_data_name_487="1026-obj-12587" 
save_data_name_488="1026-obj-12594" 
save_data_name_489="1026-obj-12597" 
save_data_name_490="1026-obj-12606" 
save_data_name_491="1026-obj-12614" 
save_data_name_492="1026-obj-12617" 
save_data_name_493="1026-obj-44781" 
save_data_name_494="1026-obj-44826" 
save_data_name_495="1026-obj-45087" 
save_data_name_496="1026-obj-45130" 
save_data_name_497="1026-obj-45164" 
save_data_name_498="1026-obj-45168" 
save_data_name_499="1026-obj-45173" 
save_data_name_500="1026-obj-45247" 
save_data_name_501="1026-obj-45261" 
save_data_name_502="1026-obj-45267" 
save_data_name_503="1026-obj-45354" 
save_data_name_504="1026-obj-45372" 
save_data_name_505="1026-obj-45524" 
save_data_name_506="1026-obj-45575" 
save_data_name_507="1026-obj-45606" 
save_data_name_508="1026-obj-45621" 
save_data_name_509="1026-obj-45638" 
save_data_name_510="1026-obj-45645" 
save_data_name_511="1026-obj-45662" 
save_data_name_512="1026-obj-45677" 
save_data_name_513="1026-obj-45699" 
save_data_name_514="1026-obj-45746" 
save_data_name_515="1026-obj-45783" 
save_data_name_516="1026-obj-45790" 
save_data_name_517="1026-obj-45822" 
save_data_name_518="1026-obj-45853" 
save_data_name_519="1026-obj-45855" 
save_data_name_520="1026-obj-45915" 
save_data_name_521="1026-obj-45964" 
save_data_name_522="1026-obj-46002" 
save_data_name_523="1026-obj-46019" 
save_data_name_524="1026-obj-46029" 
save_data_name_525="1026-obj-46033" 
save_data_name_526="1026-obj-46037" 
save_data_name_527="1026-obj-46060" 
save_data_name_528="1026-obj-46108" 
save_data_name_529="1026-obj-46117" 
save_data_name_530="1026-obj-46179" 
save_data_name_531="1026-obj-46537" 
save_data_name_532="1026-obj-46556" 
save_data_name_533="1026-obj-46598" 
save_data_name_534="1026-obj-46616" 
save_data_name_535="1026-obj-46741" 
save_data_name_536="1026-obj-46744" 
save_data_name_537="1026-obj-46847" 
save_data_name_538="1026-obj-46856" 
save_data_name_539="1026-obj-46859" 
save_data_name_540="1026-obj-46889" 
save_data_name_541="1026-obj-46955" 
save_data_name_542="1026-obj-47024" 
save_data_name_543="1026-obj-47088" 
save_data_name_544="1026-obj-47089" 
save_data_name_545="1026-obj-47252" 
save_data_name_546="1026-obj-47296" 
save_data_name_547="1026-obj-47419" 
save_data_name_548="1026-obj-47438" 
save_data_name_549="1026-obj-47601" 
save_data_name_550="1026-obj-47613" 
save_data_name_551="1026-obj-47632" 
save_data_name_552="1026-obj-47729" 
save_data_name_553="1026-obj-47853" 
save_data_name_554="1026-obj-48271" 
save_data_name_555="1026-obj-48413" 
save_data_name_556="1026-obj-48452" 
save_data_name_557="1026-obj-48467" 
save_data_name_558="1026-obj-48490" 
save_data_name_559="1026-obj-48513" 
save_data_name_560="1026-obj-48517" 
save_data_name_561="1026-obj-48721" 
save_data_name_562="1026-obj-48740" 
save_data_name_563="1026-obj-48878" 
save_data_name_564="1026-obj-7119" 
save_data_name_565="1026-obj-7167" 
save_data_name_566="1026-obj-7187" 
save_data_name_567="1026-obj-7220" 
save_data_name_568="1026-obj-7263" 
save_data_name_569="1026-obj-7290" 

torchrun --standalone --nproc_per_node=7 \
    train_ddp.py --config-name=dp3.yaml task=robogen_open_door exp_name="${exp_name}" eval_first=0  \
    use_pretrained_high_level_policy_as_low_level_input=${use_pretrained_high_level_policy_as_low_level_input} \
     task.dataset.zarr_path="[\
        ${source_dir}/${save_data_name_0},\
        ${source_dir}/${save_data_name_1},\
        ${source_dir}/${save_data_name_2},\
        ${source_dir}/${save_data_name_3},\
        ${source_dir}/${save_data_name_4},\
        ${source_dir}/${save_data_name_5},\
        ${source_dir}/${save_data_name_6},\
        ${source_dir}/${save_data_name_7},\
        ${source_dir}/${save_data_name_8},\
        ${source_dir}/${save_data_name_9},\
        ${source_dir}/${save_data_name_10},\
        ${source_dir}/${save_data_name_11},\
        ${source_dir}/${save_data_name_12},\
        ${source_dir}/${save_data_name_13},\
        ${source_dir}/${save_data_name_14},\
        ${source_dir}/${save_data_name_15},\
        ${source_dir}/${save_data_name_16},\
        ${source_dir}/${save_data_name_17},\
        ${source_dir}/${save_data_name_18},\
        ${source_dir}/${save_data_name_19},\
        ${source_dir}/${save_data_name_20},\
        ${source_dir}/${save_data_name_21},\
        ${source_dir}/${save_data_name_22},\
        ${source_dir}/${save_data_name_23},\
        ${source_dir}/${save_data_name_24},\
        ${source_dir}/${save_data_name_25},\
        ${source_dir}/${save_data_name_26},\
        ${source_dir}/${save_data_name_27},\
        ${source_dir}/${save_data_name_28},\
        ${source_dir}/${save_data_name_29},\
        ${source_dir}/${save_data_name_30},\
        ${source_dir}/${save_data_name_31},\
        ${source_dir}/${save_data_name_32},\
        ${source_dir}/${save_data_name_33},\
        ${source_dir}/${save_data_name_34},\
        ${source_dir}/${save_data_name_35},\
        ${source_dir}/${save_data_name_36},\
        ${source_dir}/${save_data_name_37},\
        ${source_dir}/${save_data_name_38},\
        ${source_dir}/${save_data_name_39},\
        ${source_dir}/${save_data_name_40},\
        ${source_dir}/${save_data_name_41},\
        ${source_dir}/${save_data_name_42},\
        ${source_dir}/${save_data_name_43},\
        ${source_dir}/${save_data_name_44},\
        ${source_dir}/${save_data_name_45},\
        ${source_dir}/${save_data_name_46},\
        ${source_dir}/${save_data_name_47},\
        ${source_dir}/${save_data_name_48},\
        ${source_dir}/${save_data_name_49},\
        ${source_dir}/${save_data_name_50},${source_dir}/${save_data_name_51},${source_dir}/${save_data_name_52},${source_dir}/${save_data_name_53},${source_dir}/${save_data_name_54},${source_dir}/${save_data_name_55},${source_dir}/${save_data_name_56},${source_dir}/${save_data_name_57},${source_dir}/${save_data_name_58},${source_dir}/${save_data_name_59}, \
        ${source_dir}/${save_data_name_60},${source_dir}/${save_data_name_61},${source_dir}/${save_data_name_62},${source_dir}/${save_data_name_63},${source_dir}/${save_data_name_64},${source_dir}/${save_data_name_65},${source_dir}/${save_data_name_66},${source_dir}/${save_data_name_67},${source_dir}/${save_data_name_68},${source_dir}/${save_data_name_69}, \
        ${source_dir}/${save_data_name_70},${source_dir}/${save_data_name_71},${source_dir}/${save_data_name_72},${source_dir}/${save_data_name_73},${source_dir}/${save_data_name_74},${source_dir}/${save_data_name_75},${source_dir}/${save_data_name_76},${source_dir}/${save_data_name_77},${source_dir}/${save_data_name_78},${source_dir}/${save_data_name_79}, \
        ${source_dir}/${save_data_name_80},${source_dir}/${save_data_name_81},${source_dir}/${save_data_name_82},${source_dir}/${save_data_name_83},${source_dir}/${save_data_name_84},${source_dir}/${save_data_name_85},${source_dir}/${save_data_name_86},${source_dir}/${save_data_name_87},${source_dir}/${save_data_name_88},${source_dir}/${save_data_name_89}, \
        ${source_dir}/${save_data_name_90},${source_dir}/${save_data_name_91},${source_dir}/${save_data_name_92},${source_dir}/${save_data_name_93},${source_dir}/${save_data_name_94},${source_dir}/${save_data_name_95},${source_dir}/${save_data_name_96},${source_dir}/${save_data_name_97},${source_dir}/${save_data_name_98},${source_dir}/${save_data_name_99}, \
        ${source_dir}/${save_data_name_100},${source_dir}/${save_data_name_101},${source_dir}/${save_data_name_102},${source_dir}/${save_data_name_103},${source_dir}/${save_data_name_104},${source_dir}/${save_data_name_105},${source_dir}/${save_data_name_106},${source_dir}/${save_data_name_107},${source_dir}/${save_data_name_108},${source_dir}/${save_data_name_109}, \
        ${source_dir}/${save_data_name_110},${source_dir}/${save_data_name_111},${source_dir}/${save_data_name_112},${source_dir}/${save_data_name_113},${source_dir}/${save_data_name_114},${source_dir}/${save_data_name_115},${source_dir}/${save_data_name_116},${source_dir}/${save_data_name_117},${source_dir}/${save_data_name_118},${source_dir}/${save_data_name_119}, \
        ${source_dir}/${save_data_name_120},${source_dir}/${save_data_name_121},${source_dir}/${save_data_name_122},${source_dir}/${save_data_name_123},${source_dir}/${save_data_name_124},${source_dir}/${save_data_name_125},${source_dir}/${save_data_name_126},${source_dir}/${save_data_name_127},${source_dir}/${save_data_name_128},${source_dir}/${save_data_name_129}, \
        ${source_dir}/${save_data_name_130},${source_dir}/${save_data_name_131},${source_dir}/${save_data_name_132},${source_dir}/${save_data_name_133},${source_dir}/${save_data_name_134},${source_dir}/${save_data_name_135},${source_dir}/${save_data_name_136},${source_dir}/${save_data_name_137},${source_dir}/${save_data_name_138},${source_dir}/${save_data_name_139}, \
        ${source_dir}/${save_data_name_140},${source_dir}/${save_data_name_141},${source_dir}/${save_data_name_142},${source_dir}/${save_data_name_143},${source_dir}/${save_data_name_144},${source_dir}/${save_data_name_145},${source_dir}/${save_data_name_146},${source_dir}/${save_data_name_147},${source_dir}/${save_data_name_148},${source_dir}/${save_data_name_149}, \
        ${source_dir}/${save_data_name_150},${source_dir}/${save_data_name_151},${source_dir}/${save_data_name_152},${source_dir}/${save_data_name_153},${source_dir}/${save_data_name_154},${source_dir}/${save_data_name_155},${source_dir}/${save_data_name_156},${source_dir}/${save_data_name_157},${source_dir}/${save_data_name_158},${source_dir}/${save_data_name_159}, \
        ${source_dir}/${save_data_name_160},${source_dir}/${save_data_name_161},${source_dir}/${save_data_name_162},${source_dir}/${save_data_name_163},${source_dir}/${save_data_name_164},${source_dir}/${save_data_name_165},${source_dir}/${save_data_name_166},${source_dir}/${save_data_name_167},${source_dir}/${save_data_name_168},${source_dir}/${save_data_name_169}, \
        ${source_dir}/${save_data_name_170},${source_dir}/${save_data_name_171},${source_dir}/${save_data_name_172},${source_dir}/${save_data_name_173},${source_dir}/${save_data_name_174},${source_dir}/${save_data_name_175},${source_dir}/${save_data_name_176},${source_dir}/${save_data_name_177},${source_dir}/${save_data_name_178},${source_dir}/${save_data_name_179}, \
        ${source_dir}/${save_data_name_180},${source_dir}/${save_data_name_181},${source_dir}/${save_data_name_182},${source_dir}/${save_data_name_183},${source_dir}/${save_data_name_184},${source_dir}/${save_data_name_185},${source_dir}/${save_data_name_186},${source_dir}/${save_data_name_187},${source_dir}/${save_data_name_188},${source_dir}/${save_data_name_189}, \
        ${source_dir}/${save_data_name_190},${source_dir}/${save_data_name_191},${source_dir}/${save_data_name_192},${source_dir}/${save_data_name_193},${source_dir}/${save_data_name_194},${source_dir}/${save_data_name_195},${source_dir}/${save_data_name_196},\
        ${source_dir2}/${save_data_name_197},${source_dir2}/${save_data_name_198},${source_dir2}/${save_data_name_199},\
        ${source_dir2}/${save_data_name_200}, ${source_dir2}/${save_data_name_201}, ${source_dir2}/${save_data_name_202}, ${source_dir2}/${save_data_name_203}, ${source_dir2}/${save_data_name_204}, ${source_dir2}/${save_data_name_205}, ${source_dir2}/${save_data_name_206}, ${source_dir2}/${save_data_name_207}, ${source_dir2}/${save_data_name_208}, ${source_dir2}/${save_data_name_209}, \
        ${source_dir2}/${save_data_name_210}, ${source_dir2}/${save_data_name_211}, ${source_dir2}/${save_data_name_212}, ${source_dir2}/${save_data_name_213}, ${source_dir2}/${save_data_name_214}, ${source_dir2}/${save_data_name_215}, ${source_dir2}/${save_data_name_216}, ${source_dir2}/${save_data_name_217}, ${source_dir2}/${save_data_name_218}, ${source_dir2}/${save_data_name_219}, \
        ${source_dir2}/${save_data_name_220}, ${source_dir2}/${save_data_name_221}, ${source_dir2}/${save_data_name_222}, ${source_dir2}/${save_data_name_223}, ${source_dir2}/${save_data_name_224}, ${source_dir2}/${save_data_name_225}, ${source_dir2}/${save_data_name_226}, ${source_dir2}/${save_data_name_227}, ${source_dir2}/${save_data_name_228}, ${source_dir2}/${save_data_name_229}, \
        ${source_dir2}/${save_data_name_230}, ${source_dir2}/${save_data_name_231}, ${source_dir2}/${save_data_name_232}, ${source_dir2}/${save_data_name_233}, ${source_dir2}/${save_data_name_234}, ${source_dir2}/${save_data_name_235}, ${source_dir2}/${save_data_name_236}, ${source_dir2}/${save_data_name_237}, ${source_dir2}/${save_data_name_238}, ${source_dir2}/${save_data_name_239}, \
        ${source_dir2}/${save_data_name_240}, ${source_dir2}/${save_data_name_241}, ${source_dir2}/${save_data_name_242}, ${source_dir2}/${save_data_name_243}, ${source_dir2}/${save_data_name_244}, ${source_dir2}/${save_data_name_245}, ${source_dir2}/${save_data_name_246}, ${source_dir2}/${save_data_name_247}, ${source_dir2}/${save_data_name_248}, ${source_dir2}/${save_data_name_249}, \
        ${source_dir2}/${save_data_name_250}, ${source_dir2}/${save_data_name_251}, ${source_dir2}/${save_data_name_252}, ${source_dir2}/${save_data_name_253}, ${source_dir2}/${save_data_name_254}, ${source_dir2}/${save_data_name_255}, ${source_dir2}/${save_data_name_256}, ${source_dir2}/${save_data_name_257}, ${source_dir2}/${save_data_name_258}, ${source_dir2}/${save_data_name_259}, \
        ${source_dir2}/${save_data_name_260}, ${source_dir2}/${save_data_name_261}, ${source_dir2}/${save_data_name_262}, ${source_dir2}/${save_data_name_263}, ${source_dir2}/${save_data_name_264}, ${source_dir2}/${save_data_name_265}, ${source_dir2}/${save_data_name_266}, ${source_dir2}/${save_data_name_267}, ${source_dir2}/${save_data_name_268}, ${source_dir2}/${save_data_name_269}, \
        ${source_dir2}/${save_data_name_270}, ${source_dir2}/${save_data_name_271}, ${source_dir2}/${save_data_name_272}, ${source_dir2}/${save_data_name_273}, ${source_dir2}/${save_data_name_274}, ${source_dir2}/${save_data_name_275}, ${source_dir2}/${save_data_name_276}, ${source_dir2}/${save_data_name_277}, ${source_dir2}/${save_data_name_278}, ${source_dir2}/${save_data_name_279}, \
        ${source_dir2}/${save_data_name_280}, ${source_dir2}/${save_data_name_281}, ${source_dir2}/${save_data_name_282}, ${source_dir2}/${save_data_name_283}, ${source_dir2}/${save_data_name_284}, ${source_dir2}/${save_data_name_285}, ${source_dir2}/${save_data_name_286}, \
        ${source_dir2}/${save_data_name_287}, ${source_dir2}/${save_data_name_288}, ${source_dir2}/${save_data_name_289}, ${source_dir2}/${save_data_name_290}, ${source_dir2}/${save_data_name_291}, ${source_dir2}/${save_data_name_292}, ${source_dir2}/${save_data_name_293}, ${source_dir2}/${save_data_name_294}, ${source_dir2}/${save_data_name_295}, ${source_dir2}/${save_data_name_296}, ${source_dir2}/${save_data_name_297}, ${source_dir2}/${save_data_name_298}, ${source_dir2}/${save_data_name_299}, ${source_dir2}/${save_data_name_300}, ${source_dir2}/${save_data_name_301}, ${source_dir2}/${save_data_name_302}, ${source_dir2}/${save_data_name_303}, ${source_dir2}/${save_data_name_304}, ${source_dir2}/${save_data_name_305}, ${source_dir2}/${save_data_name_306}, ${source_dir2}/${save_data_name_307}, ${source_dir2}/${save_data_name_308}, ${source_dir2}/${save_data_name_309}, ${source_dir2}/${save_data_name_310}, ${source_dir2}/${save_data_name_311}, ${source_dir2}/${save_data_name_312}, ${source_dir2}/${save_data_name_313}, ${source_dir2}/${save_data_name_314}, ${source_dir2}/${save_data_name_315}, ${source_dir2}/${save_data_name_316}, ${source_dir2}/${save_data_name_317}, ${source_dir2}/${save_data_name_318}, ${source_dir2}/${save_data_name_319}, ${source_dir2}/${save_data_name_320}, ${source_dir2}/${save_data_name_321}, ${source_dir2}/${save_data_name_322}, ${source_dir2}/${save_data_name_323}, ${source_dir2}/${save_data_name_324}, ${source_dir2}/${save_data_name_325}, ${source_dir2}/${save_data_name_326}, ${source_dir2}/${save_data_name_327}, ${source_dir2}/${save_data_name_328}, ${source_dir2}/${save_data_name_329}, ${source_dir2}/${save_data_name_330}, ${source_dir2}/${save_data_name_331}, ${source_dir2}/${save_data_name_332}, ${source_dir2}/${save_data_name_333}, ${source_dir2}/${save_data_name_334}, ${source_dir2}/${save_data_name_335}, ${source_dir2}/${save_data_name_336}, ${source_dir2}/${save_data_name_337}, ${source_dir2}/${save_data_name_338}, ${source_dir2}/${save_data_name_339}, ${source_dir2}/${save_data_name_340}, ${source_dir2}/${save_data_name_341}, ${source_dir2}/${save_data_name_342}, ${source_dir2}/${save_data_name_343}, ${source_dir2}/${save_data_name_344}, ${source_dir2}/${save_data_name_345}, ${source_dir2}/${save_data_name_346}, ${source_dir2}/${save_data_name_347}, ${source_dir2}/${save_data_name_348}, ${source_dir2}/${save_data_name_349}, ${source_dir2}/${save_data_name_350}, ${source_dir2}/${save_data_name_351}, ${source_dir2}/${save_data_name_352}, ${source_dir2}/${save_data_name_353}, ${source_dir2}/${save_data_name_354}, ${source_dir2}/${save_data_name_355}, ${source_dir2}/${save_data_name_356}, ${source_dir2}/${save_data_name_357}, ${source_dir2}/${save_data_name_358}, ${source_dir2}/${save_data_name_359}, ${source_dir2}/${save_data_name_360}, ${source_dir2}/${save_data_name_361}, ${source_dir2}/${save_data_name_362}, ${source_dir2}/${save_data_name_363}, ${source_dir2}/${save_data_name_364}, ${source_dir2}/${save_data_name_365}, ${source_dir2}/${save_data_name_366}, ${source_dir2}/${save_data_name_367}, ${source_dir2}/${save_data_name_368}, ${source_dir2}/${save_data_name_369}, ${source_dir2}/${save_data_name_370}, ${source_dir2}/${save_data_name_371}, ${source_dir2}/${save_data_name_372}, ${source_dir2}/${save_data_name_373}, ${source_dir2}/${save_data_name_374}, ${source_dir2}/${save_data_name_375}, ${source_dir2}/${save_data_name_376}, ${source_dir2}/${save_data_name_377}, ${source_dir2}/${save_data_name_378}, ${source_dir2}/${save_data_name_379}, ${source_dir2}/${save_data_name_380}, ${source_dir2}/${save_data_name_381}, ${source_dir2}/${save_data_name_382}, ${source_dir2}/${save_data_name_383}, ${source_dir2}/${save_data_name_384}, ${source_dir2}/${save_data_name_385}, ${source_dir2}/${save_data_name_386}, ${source_dir2}/${save_data_name_387}, ${source_dir2}/${save_data_name_388}, ${source_dir2}/${save_data_name_389}, ${source_dir2}/${save_data_name_390}, ${source_dir2}/${save_data_name_391}, ${source_dir2}/${save_data_name_392}, ${source_dir2}/${save_data_name_393}, ${source_dir2}/${save_data_name_394}, ${source_dir2}/${save_data_name_395}, ${source_dir2}/${save_data_name_396}, ${source_dir2}/${save_data_name_397}, ${source_dir2}/${save_data_name_398}, ${source_dir2}/${save_data_name_399}, ${source_dir2}/${save_data_name_400}, ${source_dir2}/${save_data_name_401}, ${source_dir2}/${save_data_name_402}, ${source_dir2}/${save_data_name_403}, ${source_dir2}/${save_data_name_404}, ${source_dir2}/${save_data_name_405}, ${source_dir2}/${save_data_name_406}, ${source_dir2}/${save_data_name_407}, ${source_dir2}/${save_data_name_408}, ${source_dir2}/${save_data_name_409}, ${source_dir2}/${save_data_name_410}, ${source_dir2}/${save_data_name_411}, ${source_dir2}/${save_data_name_412}, ${source_dir2}/${save_data_name_413}, ${source_dir2}/${save_data_name_414}, ${source_dir2}/${save_data_name_415}, ${source_dir2}/${save_data_name_416}, ${source_dir2}/${save_data_name_417}, ${source_dir2}/${save_data_name_418}, ${source_dir2}/${save_data_name_419}, ${source_dir2}/${save_data_name_420}, ${source_dir2}/${save_data_name_421}, ${source_dir2}/${save_data_name_422}, ${source_dir2}/${save_data_name_423}, ${source_dir2}/${save_data_name_424}, ${source_dir2}/${save_data_name_425}, ${source_dir2}/${save_data_name_426}, ${source_dir2}/${save_data_name_427}, ${source_dir2}/${save_data_name_428}, ${source_dir2}/${save_data_name_429}, ${source_dir2}/${save_data_name_430}, ${source_dir2}/${save_data_name_431}, ${source_dir2}/${save_data_name_432}, ${source_dir2}/${save_data_name_433}, ${source_dir2}/${save_data_name_434}, ${source_dir2}/${save_data_name_435}, ${source_dir2}/${save_data_name_436}, ${source_dir2}/${save_data_name_437}, ${source_dir2}/${save_data_name_438}, ${source_dir2}/${save_data_name_439}, ${source_dir2}/${save_data_name_440}, ${source_dir2}/${save_data_name_441}, ${source_dir2}/${save_data_name_442}, ${source_dir2}/${save_data_name_443}, ${source_dir2}/${save_data_name_444}, ${source_dir2}/${save_data_name_445}, ${source_dir2}/${save_data_name_446}, ${source_dir2}/${save_data_name_447}, ${source_dir2}/${save_data_name_448}, ${source_dir2}/${save_data_name_449}, ${source_dir2}/${save_data_name_450}, ${source_dir2}/${save_data_name_451}, ${source_dir2}/${save_data_name_452}, ${source_dir2}/${save_data_name_453}, ${source_dir2}/${save_data_name_454}, ${source_dir2}/${save_data_name_455}, ${source_dir2}/${save_data_name_456}, ${source_dir2}/${save_data_name_457}, ${source_dir2}/${save_data_name_458}, ${source_dir2}/${save_data_name_459}, ${source_dir2}/${save_data_name_460}, ${source_dir2}/${save_data_name_461}, ${source_dir2}/${save_data_name_462}, \
        ${source_dir2}/${save_data_name_463},${source_dir2}/${save_data_name_464},${source_dir2}/${save_data_name_465},${source_dir2}/${save_data_name_466},${source_dir2}/${save_data_name_467},${source_dir2}/${save_data_name_468},${source_dir2}/${save_data_name_469},${source_dir2}/${save_data_name_470},${source_dir2}/${save_data_name_471},${source_dir2}/${save_data_name_472},${source_dir2}/${save_data_name_473},${source_dir2}/${save_data_name_474},${source_dir2}/${save_data_name_475},${source_dir2}/${save_data_name_476},${source_dir2}/${save_data_name_477},${source_dir2}/${save_data_name_478},${source_dir2}/${save_data_name_479},${source_dir2}/${save_data_name_480},${source_dir2}/${save_data_name_481},${source_dir2}/${save_data_name_482},${source_dir2}/${save_data_name_483},${source_dir2}/${save_data_name_484},${source_dir2}/${save_data_name_485},${source_dir2}/${save_data_name_486},${source_dir2}/${save_data_name_487},${source_dir2}/${save_data_name_488},${source_dir2}/${save_data_name_489},${source_dir2}/${save_data_name_490},${source_dir2}/${save_data_name_491},${source_dir2}/${save_data_name_492},${source_dir2}/${save_data_name_493},${source_dir2}/${save_data_name_494},${source_dir2}/${save_data_name_495},${source_dir2}/${save_data_name_496},${source_dir2}/${save_data_name_497},${source_dir2}/${save_data_name_498},${source_dir2}/${save_data_name_499},${source_dir2}/${save_data_name_500},${source_dir2}/${save_data_name_501},${source_dir2}/${save_data_name_502},${source_dir2}/${save_data_name_503},${source_dir2}/${save_data_name_504},${source_dir2}/${save_data_name_505},${source_dir2}/${save_data_name_506},${source_dir2}/${save_data_name_507},${source_dir2}/${save_data_name_508},${source_dir2}/${save_data_name_509},${source_dir2}/${save_data_name_510},${source_dir2}/${save_data_name_511},${source_dir2}/${save_data_name_512},${source_dir2}/${save_data_name_513},${source_dir2}/${save_data_name_514},${source_dir2}/${save_data_name_515},${source_dir2}/${save_data_name_516},${source_dir2}/${save_data_name_517},${source_dir2}/${save_data_name_518},${source_dir2}/${save_data_name_519},${source_dir2}/${save_data_name_520},${source_dir2}/${save_data_name_521},${source_dir2}/${save_data_name_522},${source_dir2}/${save_data_name_523},${source_dir2}/${save_data_name_524},${source_dir2}/${save_data_name_525},${source_dir2}/${save_data_name_526},${source_dir2}/${save_data_name_527},${source_dir2}/${save_data_name_528},${source_dir2}/${save_data_name_529},${source_dir2}/${save_data_name_530},${source_dir2}/${save_data_name_531},${source_dir2}/${save_data_name_532},${source_dir2}/${save_data_name_533},${source_dir2}/${save_data_name_534},${source_dir2}/${save_data_name_535},${source_dir2}/${save_data_name_536},${source_dir2}/${save_data_name_537},${source_dir2}/${save_data_name_538},${source_dir2}/${save_data_name_539},${source_dir2}/${save_data_name_540},${source_dir2}/${save_data_name_541},${source_dir2}/${save_data_name_542},${source_dir2}/${save_data_name_543},${source_dir2}/${save_data_name_544},${source_dir2}/${save_data_name_545},${source_dir2}/${save_data_name_546},${source_dir2}/${save_data_name_547},${source_dir2}/${save_data_name_548},${source_dir2}/${save_data_name_549},${source_dir2}/${save_data_name_550},${source_dir2}/${save_data_name_551},${source_dir2}/${save_data_name_552},${source_dir2}/${save_data_name_553},${source_dir2}/${save_data_name_554},${source_dir2}/${save_data_name_555},${source_dir2}/${save_data_name_556},${source_dir2}/${save_data_name_557},${source_dir2}/${save_data_name_558},${source_dir2}/${save_data_name_559},${source_dir2}/${save_data_name_560},${source_dir2}/${save_data_name_561},${source_dir2}/${save_data_name_562},${source_dir2}/${save_data_name_563},${source_dir2}/${save_data_name_564},${source_dir2}/${save_data_name_565},${source_dir2}/${save_data_name_566},${source_dir2}/${save_data_name_567},${source_dir2}/${save_data_name_568},${source_dir2}/${save_data_name_569}\
        ]"\
    task.env_runner.demo_experiment_path="[\
        ${source_dir}/${save_data_name_0},\
        ${source_dir}/${save_data_name_1},\
        ${source_dir}/${save_data_name_2},\
        ${source_dir}/${save_data_name_3},\
        ${source_dir}/${save_data_name_4},\
        ${source_dir}/${save_data_name_5},\
        ${source_dir}/${save_data_name_6},\
        ${source_dir}/${save_data_name_7},\
        ${source_dir}/${save_data_name_8},\
        ${source_dir}/${save_data_name_9},\
        ${source_dir}/${save_data_name_10}\
    ]" \
    task.env_runner.experiment_name="[]" \
    task.env_runner.experiment_folder="[]" \
    task.env_runner.num_point_in_pc="${pointcloud_num}" \
    task.env_runner.use_absolute_waypoint="${use_absolute_waypoint}" \
    horizon="${horizon}" n_obs_steps="${n_obs_steps}" \
    task.shape_meta.obs.agent_pos.shape="[${agent_pos_dim}]" \
    task.shape_meta.action.shape="[${action_dim}]" \
    policy.pointcloud_encoder_cfg.in_channels="${pc_channel}" \
    task.dataset.observation_mode="${observation_mode}" \
    task.env_runner.observation_mode="${observation_mode}" \
    policy.encoder_type="${encoder_type}" \
    policy.encoder_output_dim=60 \
    policy.normalize_action=${normalize_action} \
    policy.act3d_encoder_cfg.in_channels=${in_channels} \
    policy.act3d_encoder_cfg.goal_mode=cross_attention_to_goal \
    policy.act3d_encoder_cfg.mode="${encoding_mode}" \
    policy.act3d_encoder_cfg.use_mlp="${use_mlp}" \
    policy.act3d_encoder_cfg.self_attention="${self_attention}" \
    policy.act3d_encoder_cfg.use_attn_for_point_features="${use_attn_for_point_features}" \
    policy.act3d_encoder_cfg.pointcloud_backbone="${pointcloud_backbone}" \
    policy.act3d_encoder_cfg.use_lightweight_unet="${use_lightweight_unet}" \
    policy.act3d_encoder_cfg.final_attention="${final_attention}" \
    task.dataset.enumerate=True \
    training.num_epochs="${training_epoches}" \
    training.rollout_every=2000 \
    training.checkpoint_every=2 \
    task.env_runner.max_steps=35 \
    task.dataset.train_ratio="${train_ratio}" \
    task.dataset.num_load_episodes=${num_load_episodes} \
    task.dataset.kept_in_disk=true \
    task.dataset.load_per_step=true \
    task.dataset.augmentation_rot="${augmentation_rot}" \
    task.dataset.augmentation_pcd="${augmentation_pcd}" \
    task.dataset.use_absolute_waypoint="${use_absolute_waypoint}" \
    task.dataset.is_pickle="${is_pickle}" \
    dataloader.batch_size="${batch_size}" \
    val_dataloader.batch_size="${batch_size}" \
    task.dataset.dataset_keys="['state', 'action', 'point_cloud', 'gripper_pcd', 'displacement_gripper_to_object', 'goal_gripper_pcd']" \
    policy.noise_model_type=transformer \
    policy.policy_type=low_level \
    training.pretrained_weighted_displacement_goal_model=/project_data/held/yufeiw2/RoboGen_sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2024-10-09_use_75_episodes_500-obj/model_57.pth \
    task.dataset.augmentation_goal_gripper_pcd=true \
    training.add_noise_to_goal_gripper_pcd=true \







    
