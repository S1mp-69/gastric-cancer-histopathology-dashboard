// Rebuild the stale editable deck using its established Calibri/teal design.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const runtime='C:/Users/gxhyp/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=runtime+'/node/node_modules';process.env.RUNTIME_NODE=runtime+'/node/bin/node.exe';process.env.UV_THREADPOOL_SIZE='2';
const {Presentation,PresentationFile}=await import(pathToFileURL(runtime+'/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs').href);
const skill='C:/Users/gxhyp/.codex/plugins/cache/openai-primary-runtime/presentations/26.1112.11217/skills/presentations';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const build=path.join(root,'.sync-backups/publication_build_20261117');
const d=JSON.parse(await fs.readFile(path.join(build,'publication_snapshot.json'),'utf8'));
const deck=Presentation.create({slideSize:{width:1280,height:720}});const tableOwners=[];
function text(s,t,x=80,y=155,w=1120,h=95,size=26,bold=false,color='#22373D'){
 const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 q.text=t;q.text.style={typeface:'Calibri',fontSize:size,bold,color,verticalAlignment:'middle',wrap:'square'};return q;
}
function slide(title,notes){const s=deck.slides.add();s.background.fill='#FAFAF7';text(s,title,80,48,1120,95,38,true,'#163F48');text(s,String(deck.slides.items.length),1185,664,45,30,18,false,'#60777A');text(s,'Amlan Choudhury | 23051241 | 17 November 2026',80,665,1000,30,17,false,'#60777A');s.speakerNotes.text=notes;return s;}
function lines(s,items,start=170,gap=115,size=28){items.forEach((t,i)=>text(s,t,85,start+i*gap,1100,gap-12,size));}
function table(s,values,widths,top=170,height=335,size=23){tableOwners.push(deck.slides.items.length);const t=s.tables.add({rows:values.length,columns:values[0].length,left:80,top,width:1120,height,values,columnWidths:widths});t.borders.assign({style:'solid',fill:'#D2DCDF',width:1});for(let r=0;r<values.length;r++)for(let c=0;c<values[r].length;c++){const cell=t.getCell(r,c);cell.fill=r===0?'#163F48':r%2?'#FFFFFF':'#EAF0EE';cell.text.style={typeface:'Calibri',fontSize:size,bold:r===0,color:r===0?'#FFFFFF':'#22373D'};}return t;}
let s=slide('Evaluating Decomposition-Guided\nStain Augmentation', 'Research evaluation draft. Verified author: Amlan Choudhury, roll 23051241, B.Tech CSE; guide Dr. Asif Uddin Khan; year supplied 2027. No author email has been supplied. Source: report_identity.json and saved experiment evidence.');
text(s,'Gastric histopathology | controlled empirical study',85,213,1100,80,30);
text(s,'Amlan Choudhury • 23051241\nB.Tech CSE | Guide: Dr. Asif Uddin Khan',85,332,1100,120,28);
text(s,'Current evidence does not establish adaptive accuracy superiority.',85,520,1100,80,28,true,'#163F48');
s=slide('Question and objectives','Question: do source diagnostics assign useful strengths beyond a matched strength bank? Stain SAN and adaptive augmentation are prior art; novelty is unconfirmed. Sources: controller.py, publication_controls.py, related-work reviews.');
lines(s,['Can decomposition diagnostics tell us which image should receive weaker stain changes?','Compare matched controls; audit data leakage; preserve reproducible outputs.','Report negative results and class failures. Keep clinical and novelty claims bounded.'],165,140);
s=slide('Two studies, different evidence limits','Sources: split_audit_20261113; prospective_grouped_manifest.csv; hiesd_expanded_v3/QC_REPORT.json. Different data, labels, model and QC prevent a causal cross-study score comparison.');
table(s,[['Study','Development data','Key limitation'],['HMU: 8 tissue classes','23,761 train / 3,059 validation','No verified patient mapping'],['HiESD: 5 tissue groups','2,543 train / 1,251 validation','Selected QC-screened subset'],['HiESD patient groups','16 observed train / 7 validation','7 additional patients reserved']], [265,425,430],180,305,24);
text(s,'The HiESD split was frozen before acquisition. It is not cross-hospital clinical validation.',85,535,1100,90,26);
s=slide('Duplicate audit changes the interpretation','Sources: manifest_exact_dedup.csv and summary.json. 932 groups/2118 files, 334 groups cross-split, 165 training-test, no cross-class exact conflicts. Exact-file control is not patient/near-duplicate control.');
table(s,[['HMU audit','Observed'],['Exact-duplicate groups / files','932 / 2,118'],['Cross-split groups','334'],['Training–test linked groups','165'],['Redundant files excluded','1,186']], [700,420],170,350,25);
text(s,'29,910 retained images. Exact byte control does not establish patient independence.',85,548,1100,75,27);
s=slide('Prior work and the contribution boundary','Sources: Kim et al. 2024 doi:10.1117/1.JMI.11.4.044006; Marini et al. 2023 doi:10.1016/j.jpi.2022.100183; Brabec et al. 2020 arxiv:2001.05571; Opitz 2024 doi:10.1162/tacl_a_00675. No numeric superiority comparison across incompatible protocols.');
table(s,[['Prior work','Established idea'],['Stain SAN; data-driven color augmentation','Stain sampling and admissible H&E variation'],['Adaptive / quality-aware augmentation','Changing perturbation strength is established'],['Brabec; Opitz','Prevalence-sensitive metrics and rankings'],['This empirical study','Matched assignment controls + integrity audit']], [500,620],175,345,23);
text(s,'No “first ever” algorithm claim. Guide and venue review must assess empirical value.',85,553,1100,65,26);
s=slide('Method: preserve the reconstruction residual','Source: controller.py and efficient_cache.py. X=BH+R; on tissue Xprime=X+alpha(sBtarget−B)H. Quantile-scaled log-condition-number and relative residual determine strength 1−0.9q. Background remains unchanged. Numerical proxies are not annotated pathology quality.');
lines(s,['Decompose optical density: X = B H + R','Perturb the stain component: X′ = X + α (sB′ − B) H','Diagnostics set α between 0.1 and 1. Residual and source identity stay attached.'],170,132,30);
text(s,'Better numerical behavior is a hypothesis—not proof of better classification.',85,565,1100,60,26,true);
s=slide('Six arms isolate what strength assignment adds','Sources: hiesd_ablation_data.py; publication_controls.py; within-class wrapper. All stain transforms precede shared geometry/RGB pipeline. Within-class shuffle is a training-label control, not deployment policy.');
table(s,[['Arm','Assignment'],['Standard','Shared geometry + RGB augmentation only'],['Residual fixed','Additional residual-preserving stain, α = 1'],['Uniform weaker','One mean training strength, about 0.6135'],['Global shuffled','Same overall strength bank; random source assignment'],['Adaptive','Each source receives its diagnostic strength'],['Within-class shuffled','Same strength bank per tissue class; random sources']], [350,770],156,425,22);
text(s,'The added within-class control is post-hoc; no parameters are tuned for it.',85,594,1100,40,24);
s=slide('Backbones and matched training','Sources: app/ml/model.py, hiesd_resnet18.py, hiesd_ablation_train.py and config files. ResNet wrapper name is historical; actual network is torchvision ResNet18 with no LSTM.');
table(s,[['Component','HMU / dashboard','HiESD research'],['Backbone','CNN → BiLSTM → attention','ImageNet-pretrained ResNet18'],['Output labels','8 tissue classes','5 mapped tissue groups'],['Replication','3 seeds per original arm','3 seeds per arm'],['Stop rule','40-epoch cap, patience 7','40-epoch cap, patience 7']], [270,425,425],172,330,23);
text(s,'Same seed → same initial weights, split, geometry, order and optimizer within each study.',85,540,1100,80,27);
s=slide('Efficient, auditable execution','Sources: HiESD config/histories; cache reference; publication_campaign_audit.py. HiESD uses batch16, workers0, threads2, FP16. Best+last state each epoch. Earlier HMU benchmark 238.7 images/s is not HiESD throughput. No clock/thermal changes.');
lines(s,['Versioned source decompositions are cached; random target stains remain random.','HiESD: batch 16, 2 CPU threads, FP16; resumable best and last checkpoints.','All 15 original HiESD runs passed checkpoint, split and saved-prediction audits.'],170,135);
text(s,'Local supervision continues without repeated ChatGPT polling.',85,579,1100,50,27,true);
s=slide('HMU validation: shuffled is ahead on average','Sources: final_campaign_20261113 and shuffled_20261116. Mean macro-F1 three seeds; adaptive vs shuffled exploratory interval includes zero. Image split does not establish patient independence.');
table(s,[['Arm','Mean validation macro-F1'],...d.hmu.map(r=>[r.label,r.mean.toFixed(4)])],[650,470],170,350,26);
text(s,'Adaptive − global shuffled: −0.0054. Adaptive tumor recall is lower than standard.',85,548,1100,78,27);
s=slide('HiESD validation: no adaptive advantage','Source: hiesd_resnet_ablation_v1/summary.json. Three repeated seeds on same1251validation cases. Error bars in manuscript are seed SD, not patient uncertainty. Adaptive minus shuffled seed differences −.0273,+.0008,−.0391.');
table(s,[['Arm','Seeds','Macro-F1 mean ± SD'],...d.hiesd.filter(r=>r.arm!=='within_class_shuffled').map(r=>[r.label,String(r.n),r.macro_f1_mean.toFixed(4)+' ± '+r.macro_f1_sd.toFixed(4)])],[430,160,530],166,360,25);
text(s,'Adaptive − global shuffled: −0.0219 across the three seeds.',85,555,1100,65,28,true);
s=slide('Added within-class control','Source: hiesd_within_class_v1/protocol.json, wrapper_provenance.json and RESULTS.json. Conditional permutation is post-hoc falsification; same frozen trainer and initial weights. No proposal search is launched.');
const w=d.hiesd.find(r=>r.arm==='within_class_shuffled');
lines(s,['Purpose: keep each tissue class strength distribution fixed while breaking source assignment.',d.within_status+'.',w.n===3?'Completed mean macro-F1: '+w.macro_f1_mean.toFixed(4)+' ± '+w.macro_f1_sd.toFixed(4):'Partial runs are not presented as a three-seed conclusion.'],165,135);
text(s,'A stronger control can challenge the hypothesis even when the method fails.',85,580,1100,50,26);
s=slide('Frozen reserved-patient test','Sources: final_test_hiesd_v1/PROTOCOL.json, checkpoint freeze, QC and RESULTS if complete. The primary contrast was fixed before reserved pixels. No test tuning. Seven patients make cluster intervals descriptive/unstable.');
if(d.test_complete){table(s,[['Arm','Mean test macro-F1'],...Object.entries(d.test.arm_means).map(([a,v])=>[d.labels[a],v.macro_f1.toFixed(4)])],[700,420],160,365,23);text(s,`Primary adaptive − within-class: ${d.test.primary_adaptive_minus_within_class.toFixed(4)}. ${d.test.patches} patches / ${d.test.patients} patients.`,85,548,1100,80,26);}
else{lines(s,['13 reserved slides from 7 patients; about 7.53 GiB of source data.','Primary contrast: adaptive minus within-class shuffled mean patch macro-F1.',d.failure?'Blocked: '+d.failure.error:'Current stage: '+(d.final_status.stage||'waiting for control completion')+'. No final-test scores yet.'],163,135,27);text(s,'All six arms and three seeds are frozen before test access. No subsequent tuning.',85,581,1100,50,25,true);}
s=slide('Failures and inference limits','Sources: per_class_details in HiESD summary; mask/QC receipts; publication audits. Only seven lymphoid validation patches. Other recall poor. Selected subset is not clinical cohort, no local expert adjudication. Historical demo prototype overlap disclosed.');
lines(s,['Only 7 validation lymphoid patches; the heterogeneous “other” class has weak recall.','Coarse masks and annotation-driven sampling limit interpretation.','Public-release patient holdout is not independent hospital or clinical validation.'],160,135);
text(s,'Numerical stability ≠ classifier validation ≠ historical notebook performance.',85,580,1100,50,27,true);
s=slide('Secondary reweighting: known metric behavior','Sources: ranking_audit_20261117 verification; Brabec2020; Opitz2024. 24 correlated recipes,22/23 winnerchanges21/23contrastsignchanges. Lymphoid driver69.4%. Exact reweighting fixed predictions, not QC intervention or clinical prevalence.');
lines(s,['24 hypothetical class/patient weighting recipes on the same saved predictions.','22/23 alternative winners change; 21/23 adaptive–shuffled signs change.','This illustrates known F1 prevalence sensitivity. No new metric is claimed.'],165,135);
text(s,'No retraining under alternative QC. No assumed clinical prevalence.',85,580,1100,50,27,true);
s=slide('Implementation and demonstration','Sources: app/main.py, app/ml/inference.py, app/db/models.py and demo_serve.py. Wrapper uses separate DB/uploads, CPU, train-only prototypes. Do not activate research checkpoints.');
table(s,[['Layer','Technology / role'],['Interface','HTML, CSS, JavaScript; image upload and history'],['API and storage','FastAPI; SQLAlchemy; SQLite prediction_records'],['Research','Python, PyTorch, torchvision, NumPy, OpenSlide'],['Demo safeguards','Isolated DB; training-only references; original model']], [350,770],172,345,25);
text(s,'Show the returned model_source. A prototype fallback is not the research ResNet18.',85,551,1100,78,27);
s=slide('What can be concluded?','Sources: both original three-seed studies; final test snapshot; train-only assignment feasibility/control. No accuracy-superiority or guaranteed novelty claim. A rigorous negative empirical study may be considered depending on adequacy and venue.');
lines(s,['The completed original experiments do not support diagnostic-guided accuracy benefit.','The useful outcome is a controlled, reproducible test of a plausible hypothesis.','Publication value depends on full evidence, limits and guide/venue review.'],170,135);
s=slide('Submission package and remaining requirements','Sources: current evaluation_package files. Real Overleaf login/project, genuine Turnitin reports, author/ethics/funding/conflict confirmation and guide signoff are external. Teacher thresholds <10%similarity,<20%AI are requirements not guaranteed scores. No submission/payment/external message.');
lines(s,['Prepared: prescribed-format report, IEEE source/PDF, editable slides and evidence.','Next: guide critique; authentic Overleaf link; genuine institutional Turnitin reports.','Confirm declarations, authorship and venue fit. No guaranteed novelty or acceptance.'],170,130);
text(s,`Evidence snapshot: ${d.as_of}`,85,579,1100,50,23);
const candidate=path.join(build,'publication_deck_candidate.pptx');await (await PresentationFile.exportPptx(deck)).save(candidate);
const finalDir=path.join(build,'validated');await fs.mkdir(finalDir,{recursive:true});const final=path.join(finalDir,'publication_'+Date.now()+'.pptx');
const reference=path.join(root,'.sync-backups/evaluation_package_original_20261113/Progress_Presentation.pptx');
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:final,pythonExecutable:runtime+'/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],explicitTotalSlideCount:18,requiredNativeTableOwnerSlides:tableOwners,fontPolicy:{basis:'reference',families:['Calibri'],referencePath:reference,referenceSha256:createHash('sha256').update(await fs.readFile(reference)).digest('hex')},verifyArtifactToolImport:true,receiptPath:path.join(build,'deck_validation_'+Date.now()+'.json')});
await fs.copyFile(final,path.join(build,'Progress_Presentation.pptx'));
for(let i=0;i<deck.slides.items.length;i++){const blob=await deck.export({slide:deck.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(build,`slide_${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));}
await fs.writeFile(path.join(build,'deck_layout.ndjson'),(await deck.inspect({kind:'slide,textbox,table,notes',maxChars:250000})).ndjson);
console.log(JSON.stringify({slides:deck.slides.items.length,tableOwners,snapshot:d.as_of}));
