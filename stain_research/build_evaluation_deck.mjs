// Update the transferred editable deck with actual overnight status.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const runtime='C:/Users/gxhyp/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=runtime+'/node/node_modules';
process.env.RUNTIME_NODE=runtime+'/node/bin/node.exe';
process.env.UV_THREADPOOL_SIZE='2';
const {PresentationFile,FileBlob}=await import(pathToFileURL(runtime+'/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs').href);
const skill='C:/Users/gxhyp/.codex/plugins/cache/openai-primary-runtime/presentations/26.1111.10930/skills/presentations';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const build=path.join(root,'.sync-backups/evaluation_build');
await fs.mkdir(build,{recursive:true});
const snapshot=JSON.parse(await fs.readFile(path.join(build,'evaluation_snapshot.json'),'utf8'));
const deck=await PresentationFile.importPptx(await FileBlob.load(path.join(root,'.sync-backups/evaluation_package_original_20261113/Progress_Presentation.pptx')));
const oldInspect=await deck.inspect({kind:'slide,textbox,table,notes',maxChars:200000});
await fs.writeFile(path.join(build,'before_layout.ndjson'),oldInspect.ndjson);
// Clarify that the old six-arm table remains the three-epoch pilot.
for(const item of (await deck.inspect({kind:'textbox',maxChars:100000})).ndjson.split('\n').filter(Boolean).map(v=>JSON.parse(v))){
 if(item.text==='Classifier validation results') deck.resolve(item.id).text='Three-epoch pilot validation';
 if(item.text==='Progress and next experiments') deck.resolve(item.id).text='Progress and current campaign';
 if(item.text?.startsWith('Completed: application, controller,')) deck.resolve(item.id).text='Completed: application, controller, numerical checks and duplicate audit.';
 if(item.text?.includes('converged baseline') || item.text?.includes('Next: converged')) deck.resolve(item.id).text='Completed: four augmentation arms, three matched seeds each.';
 if(item.text?.startsWith('Amlan |')) deck.resolve(item.id).text='Amlan Choudhury | 23051241 | 13 November 2026';
 if(item.text?.startsWith('Compare weaker')) deck.resolve(item.id).text='Adaptive mean macro-F1: 0.6836; fixed: 0.6822.';
 if(item.text?.startsWith('Repeat seeds')) deck.resolve(item.id).text='Mean tumor recall: adaptive 75.56%; standard 79.97%.';
}
function textbox(slide,text,x,y,w,h,size=25,bold=false,color='#22373D'){
 const item=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 item.text=text;item.text.style={typeface:'Calibri',fontSize:size,bold,color,verticalAlignment:'middle',wrap:'square'};
 return item;
}
function slide(title,after){
 const item=deck.slides.insert({after,background:{fill:'#FAFAF7'}}).slide;
 textbox(item,title,80,55,1120,75,38,true,'#163F48');return item;
}
function table(slide,values,widths,top=176,height=270){
 const t=slide.tables.add({rows:values.length,columns:values[0].length,left:80,top,width:1120,height,values,columnWidths:widths});
 t.borders.assign({style:'solid',fill:'#D2DCDF',width:1});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[r].length;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?'#163F48':(r%2?'#FFFFFF':'#EAF0EE');
  cell.text.style={typeface:'Calibri',fontSize:23,bold:r===0,color:r===0?'#FFFFFF':'#22373D'};
 }
 return t;
}
const status=slide('Completed three-seed validation',11);
textbox(status,`Snapshot ${snapshot.as_of_display}`,84,136,1110,34,20);
const labels={standard:'Standard',residual_fixed:'Residual fixed',uniform_weaker:'Uniform weaker',adaptive:'Adaptive'};
const rows=[['Arm','Accuracy mean','Macro-F1 mean ± SD','TUM recall'],...snapshot.final_analysis.summary.map(r=>[
 labels[r.arm],(100*r.accuracy_mean).toFixed(2)+'%',r.macro_f1_mean.toFixed(4)+' ± '+r.macro_f1_sd.toFixed(4),(100*r.recall_mean.TUM).toFixed(2)+'%'])];
table(status,rows,[270,240,360,250],188,260);
textbox(status,'Adaptive − fixed paired F1: +0.0227, −0.0101, −0.0084. No established superiority.',86,480,1100,65,24);
textbox(status,'Same 3,059 validation images across seeds; patient independence unverified.',86,563,1100,65,24);
status.speakerNotes.text='This is the new overnight campaign, separate from the earlier three-epoch pilot. Each arm shares its seed-specific initialization, split, geometry, optimizer, 40-epoch ceiling and early stopping rule. Values shown for active runs are provisional best-so-far validation results, not independent test results. No test inference is used to select methods. Source: evaluation_snapshot.json and actual run/history.json / metrics.json.';
const compute=slide('Measured throughput and restartable training',12);
textbox(status,'13',1180,666,40,30,18,false,'#60777A');
textbox(compute,'14',1180,666,40,30,18,false,'#60777A');
table(compute,[['Setting','Observed result'],
 ['Cached augmentation, batch 64, four workers',snapshot.throughput.toFixed(1)+' images/second'],
 ['Batch 96', 'Rejected for insufficient VRAM headroom'],
 ['Exact deterministic cache checks','64 / 64 pixel-identical'],
 ['Cache extraction failures','2 images retained with unchanged fallback']], [670,450],174,270);
textbox(compute,'Main CPU threads: 2. Each loader worker: 1. Mixed precision uses FP16.',86,478,1100,65,24);
textbox(compute,'Resumable state saves every 120 seconds and each epoch. The 9 AM snapshot does not stop training.',86,564,1100,85,24);
compute.speakerNotes.text='The benchmark used training images, cached residual augmentation, geometry, host-to-device transfer and actual GPU optimizer steps. It discarded all benchmark weights. Batch 64 left approximately 1.2 GiB CUDA free in the measured trial. Batch 96 was slightly faster but failed the headroom rule. Actual epoch timings and GPU telemetry remain in the campaign logs. The float64 cache does not freeze stochastic target stains, intensity changes or geometry. Two extraction failures occur in ADI training images. Source: throughput_benchmark.json, cpu_benchmark.json, cache_equivalence.json and cache failure records.';
for(let i=0;i<deck.slides.items.length;i++){
 const s=deck.slides.items[i];
 const records=(await deck.inspect({kind:'textbox',maxChars:100000})).ndjson.split('\n').filter(Boolean).map(v=>JSON.parse(v));
 for(const r of records)if(r.slideIndex===i && r.position?.left>=1170 && r.position?.top>=660 && /^\d+$/.test(r.text||'')) deck.resolve(r.id).text=String(i+1);
}
const candidate=path.join(build,'deck_candidate.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidate);
await fs.mkdir(path.join(build,'validated'),{recursive:true});
const final=path.join(build,'validated','deck_validated_'+Date.now()+'.pptx');
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:final,
 pythonExecutable:runtime+'/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',
             ...[5,6,8,10,13,14].flatMap(n=>['--require-native-table-slide',String(n)])],
 explicitTotalSlideCount:15,requiredNativeTableOwnerSlides:[5,6,8,10,13,14],
 fontPolicy:{basis:'reference',families:['Calibri'],referencePath:path.join(root,'.sync-backups/evaluation_package_original_20261113/Progress_Presentation.pptx'),referenceSha256:createHash('sha256').update(await fs.readFile(path.join(root,'.sync-backups/evaluation_package_original_20261113/Progress_Presentation.pptx'))).digest('hex')},
 verifyArtifactToolImport:true,receiptPath:path.join(build,'deck_validation_'+Date.now()+'.json')});
for(let i=0;i<deck.slides.items.length;i++){
 const image=await deck.export({slide:deck.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(build,`final_slide_${i+1}.png`),new Uint8Array(await image.arrayBuffer()));
}
await fs.writeFile(path.join(build,'after_layout.ndjson'),(await deck.inspect({kind:'slide,textbox,table,notes',maxChars:200000})).ndjson);
await fs.copyFile(final,path.join(build,'Progress_Presentation.pptx'));
console.log(JSON.stringify({slides:deck.slides.items.length,as_of:snapshot.as_of_display,pptx:'Progress_Presentation.pptx',pdf:'Progress_Presentation.pdf'}));
