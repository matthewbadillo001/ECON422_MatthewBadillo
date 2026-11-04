// Start the studio and pass its local URL: node tests/test_writing_hints.cjs http://127.0.0.1:PORT
const assert = require('node:assert/strict');
const vm = require('node:vm');

async function test() {
    const html = await (await fetch(process.argv[2])).text();
    const elements = {};
    let focused = null;
    function element(id) {
        const classes = new Set();
        return {id, hidden:false, checked:false, value:'', files:[], children:[], dataset:{},
            classList:{toggle(name,on){if(on) classes.add(name); else classes.delete(name)},contains:name=>classes.has(name)},
            setCustomValidity(){}, setAttribute(k,v){this[k]=v},
            addEventListener(k,callback){this['on'+k]=callback},
            replaceChildren(){this.children=[]}, append(...nodes){this.children.push(...nodes)},
            contains(target){return this===target || this.children.some(child=>child.contains(target))},
            focus(){if(focused!==this){focused=this;this.onfocus?.()}},
            querySelector(){return this.small || (this.small=element())}};
    }
    for (const match of html.matchAll(/\bid="([^"]+)"/g)) elements[match[1]]=element(match[1]);
    const fields = [...html.matchAll(/<textarea id="(\w+)" data-criterion="(\w+)"/g)].map(m=>{
        const field=elements[m[1]];field.dataset.criterion=m[2]; return field;
    });
    elements.liveHints.checked = /id="liveHints" checked/.test(html);
    const requests=[], timers=new Map();
    let timerCount=0;
    const tick=()=>{const next=[...timers.values()];timers.clear();for(const cb of next)cb()};
    const settle=()=>new Promise(resolve=>setImmediate(resolve));
    const document={getElementById:id=>elements[id],querySelectorAll:()=>fields,createElement:()=>element(),
        get activeElement(){return focused}};
    let sourceFetch;
    function fetchHint(path, options){
        if (sourceFetch) return sourceFetch(path, options);
        if(path==='/analyze') return fetch(process.argv[2]+path,options);
        assert.equal(path,'/writing-hints');
        const body=JSON.parse(options.body);
        return new Promise(resolve=>requests.push({body,respond:wording=>resolve({
            ok:true,json:async()=>({source:'Reviewer suggestions',
                candidates:Object.fromEntries(body.targets.map(key=>[key,[wording+' '+key]]))})})}));
    }
    const stored=new Map();
    const tabStorage={getItem:key=>stored.get(key) ?? null,setItem:(key,val)=>stored.set(key,val),removeItem:key=>stored.delete(key)};
    const boot=storage=>vm.runInNewContext(html.match(/<script>([\s\S]+)<\/script>/)[1],
        {document, URL, TextDecoder, sessionStorage:storage, fetch:fetchHint, setTimeout:cb=>{timers.set(++timerCount,cb);return timerCount},
         clearTimeout:key=>timers.delete(key)});
    boot(tabStorage);
    const e=elements, panel=id=>e['help-'+id];
    const input=(id,text)=>{e[id].focus();e[id].value=text;e[id].oninput()};
    const choice=id=>panel(id).children.find(b=>b.className==='candidate');
    const marked=id=>e[id].classList.contains('needs-detail') && !e['empty-'+id].hidden;

    assert(html.includes('<bdi id="labOwner"></bdi>LOCAL ECONOMETRICS LAB'));
    assert.equal(e.labOwner.textContent,'');
    const promptBeforeName=e.promptPreview.textContent;
    assert(html.includes('id="labNameEditor" hidden'));
    e.labHeader.children.push(e.editLabName,e.labNameEditor);e.labNameEditor.children.push(e.labName);
    const focusBeforeHover=focused;
    e.labHeader.onpointerenter({pointerType:'mouse'});assert(!e.labNameEditor.hidden);assert.equal(focused,focusBeforeHover);
    e.labHeader.onpointerleave();assert(e.labNameEditor.hidden);
    e.labHeader.onpointerenter({pointerType:'touch'});assert(e.labNameEditor.hidden);
    e.editLabName.onclick();assert.equal(focused,e.labName);e.labHeader.onpointerleave();assert(!e.labNameEditor.hidden);
    e.labHeader.onkeydown({key:'Escape',target:e.labName,preventDefault(){}});assert(e.labNameEditor.hidden);assert.equal(focused,e.editLabName);
    e.editLabName.onfocus();assert(!e.labNameEditor.hidden);assert.equal(e.editLabName['aria-expanded'],'true');
    e.labHeader.onfocusout({relatedTarget:e.labName});assert(!e.labNameEditor.hidden);
    e.labHeader.onkeydown({key:'Enter',target:e.labName,preventDefault(){}});assert(e.labNameEditor.hidden);
    e.editLabName.onclick();e.labHeader.onfocusout({relatedTarget:e.clarity});assert(e.labNameEditor.hidden);
    input('labName','  Jieun Lee  ');assert.equal(e.labOwner.textContent,'Jieun Lee’s ');
    e.labName.value='';boot(tabStorage);assert.equal(e.labName.value,'Jieun Lee');
    assert.equal(e.labOwner.textContent,'Jieun Lee’s ');assert.equal(e.promptPreview.textContent,promptBeforeName);
    input('labName',"Élodie O’Brien");assert.equal(e.labOwner.textContent,"Élodie O’Brien’s ");
    input('labName','  ');boot(tabStorage);assert.equal(e.labOwner.textContent,'');assert.equal(stored.size,0);
    boot(undefined);input('labName','이');assert.equal(e.labOwner.textContent,'이’s ');
    input('labName','');assert.equal(e.promptPreview.textContent,promptBeforeName);
    console.log('Passed: optional lab name, refresh, clearing, and storage-disabled use.');
    assert(!e['division-prompt'] && !e['category-real']);
    assert.equal(e['division-forecasting']['aria-selected'],'true');
    assert.equal(e['category-forecasting-blank']['aria-pressed'],'true');
    assert(fields.length===5 && fields.every(f=>f.value===''));assert(e.run.disabled);
    e['division-estimation'].onclick();
    assert.equal(e['category-estimation-blank']['aria-pressed'],'true');
    assert.equal((html.match(/>Prompt writing<\/button>/g)||[]).length,2);
    const estShelf=html.split('id="shelf-estimation-blank"')[1].split('id="shelf-regression"')[0];
    assert(estShelf.includes('<summary>Try a writing exercise</summary>'));
    assert(fields.every(f=>!f.value));
    e.estimationIncomplete.onclick();
    assert(e.clarity.value.includes('training_hours') && e.context.value.includes('education'));
    assert.equal(e.dataKind.value,'cross_section');assert.equal(e.sourceMode.value,'local');
    assert.deepEqual(fields.filter(f=>!f.value).map(f=>f.id),['structure','role','examples']);
    e.estimationRough.onclick();assert(fields.every(f=>f.value) && e.structure.value.includes('DML'));
    e['division-forecasting'].onclick();e['division-estimation'].onclick();

    assert(fields.every(f=>f.value===''));assert(e.reviewDraft.disabled);
    assert(!e['category-joint'] && !e['category-sources'] && !e.joint_var && !e.fredExample);
    e.liveHints.checked=false; e.liveHints.onchange();
    input('clarity','Estimate the difference in earnings for an extra hour of training.');
    const original=fields.map(f=>f.value);
    let onReview, seen=0;
    sourceFetch=(path,options)=>{
        assert.equal(path,'/practice-review');seen++;
        const body=JSON.parse(options.body);assert.deepEqual(Object.keys(body),['prompt_fields']);
        return new Promise(resolve=>{onReview=()=>resolve(Response.json({
            source:'Local Llama writing feedback',notes:Object.fromEntries(fields.map(f=>[f.id,'Review '+f.id]))}));});
    };
    const first=e.reviewDraft.onclick();onReview();await first;
    assert.equal(seen,1);assert.equal(e.practiceNotes.children.length,5);assert.deepEqual(fields.map(f=>f.value),original);
    const later=e.reviewDraft.onclick();
    input('context','Use one row per worker.');onReview();await later;assert(e.practiceNotes.hidden);
    input('structure','Estimate the adjusted coefficient and report a confidence interval.');
    input('role','Explain the estimate to an economics student.');
    input('examples','Interpret one additional hour of training.');
    const handwritten=Object.fromEntries(fields.map(f=>[f.id,f.value]));
    e['category-cross'].onclick();
    assert.deepEqual(Object.fromEntries(fields.map(f=>[f.id,f.value])),handwritten);
    assert(e.analysisWorkspace.hidden && !e.analysisControls.hidden && !e.run.disabled && e.reference.hidden);
    const trainingCsv='earnings,training_hours\n1000,2';
    e.dataKind.value='cross_section';e.sourceMode.value='local';
    e.file.files=[{name:'workers.csv',size:trainingCsv.length,text:async()=>trainingCsv}];
    sourceFetch=(path,opts)=>{
        assert.equal(path,'/analyze');
        const body=JSON.parse(opts.body);
        assert.equal(body.mode,'live');assert.deepEqual(body.prompt_fields,handwritten);
        assert.equal(body.data_kind,'cross_section');assert.equal(body.filename,'workers.csv');assert.equal(body.csv,trainingCsv);
        return new Response(JSON.stringify({type:'result',report:{id:'real-test',version:1,task:'cross_dml',
            summary:'Adjusted earnings estimate',columns:['Term'],rows:[['training_hours']],audit:{}}})+'\n');
    };
    await e.form.onsubmit({preventDefault(){},submitter:e.run});
    assert.equal(e.summary.textContent,'Adjusted earnings estimate');
    assert.equal(e['division-estimation']['aria-selected'],'true');
    e['category-estimation-blank'].onclick();
    assert(fields.every(f=>f.value==='') && e.run.disabled && e.reviewDraft.disabled);
    assert.equal(e.dataKind.value,'auto');assert.equal(e.sourceMode.value,'auto');assert.equal(e.file.value,'');
    e.file.files=[];
    sourceFetch=(path,opts)=>fetch(process.argv[2]+path,opts);
    e['category-cross'].onclick();await e.crossExample.onclick();
    assert(fields.every(f=>f.value));
    e['division-estimation'].onkeydown({key:'ArrowRight',preventDefault(){}});
    assert.equal(e['division-forecasting']['aria-selected'],'true');
    assert(fields.every(f=>f.value==='') && e['category-forecasting-blank']['aria-pressed']==='true');
    e['category-hybrid'].onclick();assert(fields.every(f=>f.value===''));
    await e.seasonForest.onclick();assert(fields.every(f=>f.value));
    e['category-forecasting-blank'].onclick();assert(fields.every(f=>f.value===''));
    input('clarity','My own monthly forecast');e['division-forecasting'].onclick();
    assert.equal(e.clarity.value,'My own monthly forecast');
    e['division-estimation'].onclick();assert(fields.every(f=>f.value===''));
    e['division-forecasting'].onclick();e['category-hybrid'].onclick();await e.seasonForest.onclick();
    sourceFetch=undefined;e.liveHints.checked=true; e.liveHints.onchange();
    console.log('Passed: two workspaces open blank, manual prompts can be reviewed and run, browsing keeps drafts, and templates load only by choice.');
    assert.equal(e.reference.disabled,false);
    input('structure','   ');
    assert(marked('structure'));
    assert.equal(e.fieldNotice.textContent,'Please complete Structure.');
    tick();
    assert.equal(requests.length,1);
    assert.deepEqual(requests[0].body.targets,['structure']);
    assert(choice('structure'));
    assert.equal(e.structure.value,'   ');

    input('clarity','Forecast next month');
    input('clarity','Forecast fare inflation next month.');
    tick(); // A new request waits for the old one to finish.
    assert.equal(requests.length,1);
    requests[0].respond('Old draft');await settle();
    assert.equal(requests.length,2);
    assert.deepEqual(requests[1].body.targets,['clarity','structure']);
    assert.equal(requests[1].body.prompt_fields.clarity,e.clarity.value);
    assert(!choice('structure').textContent.includes('Old draft'));
    requests[1].respond('Latest wording for');await settle();
    assert.equal(choice('structure').textContent,'Latest wording for structure');
    assert.equal(focused,e.clarity);
    assert.equal(e.structure.value,'   ');
    const before=fields.map(f=>f.value);
    choice('structure').onclick(); // Accept only this field.
    assert.equal(e.structure.value,'Latest wording for structure');
    assert(!marked('structure'));
    assert(e.fieldNotice.hidden);
    for (let i of [0,2,3,4]) assert.equal(fields[i].value,before[i]);

    input('structure','First fit on earlier months.');
    tick();requests[2].respond('Then score');await settle();
    assert.equal(e.structure.value,'First fit on earlier months.');
    choice('structure').onclick();
    assert.equal(e.structure.value,'First fit on earlier months.\nThen score structure');

    e.incomplete.onclick();tick();
    assert.deepEqual(requests[3].body.targets,['structure','role','examples']);
    for (const key of ['structure','role','examples']) assert(marked(key) && !panel(key).hidden);
    await e.estimate.onclick();requests[3].respond('Stale example');await settle();
    assert(fields.every(f=>panel(f.id).hidden && !marked(f.id)));
    assert.equal(requests.length,4);

    input('structure','');tick();
    requests[4].respond('Choose this');await settle();
    panel('structure').children.at(-1).onclick(); // Write my own suppresses this box's suggestions.
    input('structure','My own first step.');tick();
    assert(panel('structure').hidden);
    assert.equal(requests.length,5);
    e.liveHints.checked=false;e.liveHints.onchange();
    input('role','');tick();
    assert(marked('role') && panel('role').hidden);
    assert.equal(requests.length,5);

    e.liveHints.checked=true;e.liveHints.onchange();tick();
    e.liveHints.checked=false;e.liveHints.onchange();
    requests[5].respond('Do not reopen');await settle();
    assert(panel('role').hidden);
    assert(marked('role'));
    const key='ab'.repeat(16), csv='date,UNRATE\n2023-01-01,3.4\n';
    const analysisBodies=[], credentials=[];
    sourceFetch=async(path,opts)=>{
        const body=JSON.parse(opts.body);
        if(path==='/sources/fred'){
            credentials.push(body);
            if(credentials.length===1) return Response.json({error:'Check the key and series ID.'},{status:400});
            return Response.json({csv,filename:'FRED monthly data (UNRATE)',series:['UNRATE']});
        }
        assert.equal(path,'/analyze');
        assert(!opts.body.includes(key));
        analysisBodies.push(body);
        const event = !body.csv ? {type:'needs_source',message:'Connect to FRED.'} :
            !body.clarifications ? {type:'needs_input',question:'Which series should I forecast?'} :
            {type:'result',report:{id:'run1',version:1,source:'FRED',task:'forecast',columns:[],rows:[],audit:{}}};
        return new Response(JSON.stringify(event)+'\n');
    };
    e['division-forecasting'].onclick();
    e['category-forecasting-blank'].onclick();
    for(const f of fields) input(f.id,f.id==='clarity' ? 'Forecast the monthly unemployment rate UNRATE.' : 'Use monthly U.S. unemployment observations.');
    e.sourceMode.value='fred';e.sourceMode.onchange();
    const submit=()=>e.form.onsubmit({preventDefault(){}, submitter:e.run});
    await submit();assert.equal(e.sourcePanel.hidden,false);assert.equal(focused,e.fredIds);e.fredIds.value='UNRATE';
    assert.equal(e.sourceKey.value,'');
    assert(/type="password" id="sourceKey"/.test(html));
    e.sourceKey.value=key;await e.connectSource.onclick();
    assert.equal(e.sourceKey.value,'');assert(!e.connectSource.disabled);assert(!e.sourcePanel.hidden);
    assert(e.sourceStatus.textContent.includes('Check the key'));
    e.sourceKey.value=key;await e.connectSource.onclick();
    assert.equal(e.sourceKey.value,'');assert(e.sourcePanel.hidden);assert(!e.saveInput.hidden);
    assert.equal(e.clarificationPanel.hidden,false);
    e.clarificationAnswer.value='UNRATE';await e.continueReview.onclick();
    assert.equal(e.results.hidden,false);assert.equal(analysisBodies[2].csv,csv);
    assert.equal(analysisBodies[2].clarifications[0].answer,'UNRATE');
    assert.deepEqual(credentials,[{series_ids:'UNRATE',api_key:key},{series_ids:'UNRATE',api_key:key}]);
    const webRequests=[];
    sourceFetch=async(path,opts)=>{
        const body=JSON.parse(opts.body);
        if(path==='/sources/web'){
            webRequests.push(body);
            return Response.json({csv:'date,electricity_price\n2023-01-01,10.5\n',filename:'fixture monthly data'});
        }
        assert.equal(path,'/analyze');assert(!opts.body.includes(key));
        if(!body.csv) return fetch(process.argv[2]+path,opts);
        return new Response(JSON.stringify({type:'result',report:{id:'run2',version:1,task:'forecast',columns:[],rows:[],audit:{}}})+'\n');
    };
    e.sourceMode.value='eia';e.sourceMode.onchange();
    input('clarity','Forecast electricity_price using its available monthly lags.');
    input('context','Use the Colorado electricity price in cents per kilowatthour.');await submit();
    assert.equal(e.sourceTitle.textContent,'Connect to U.S. Energy Information Administration (EIA)');
    assert(e.fredFields.hidden && !e.webFields.hidden);assert.equal(e.apiPath.value,'response.data');
    assert.equal(e.apiValues.value,'electricity_price=price');assert.equal(e.apiParam.value,'api_key');
    e.sourceKey.value=key;await e.connectSource.onclick();assert.equal(webRequests[0].provider,'eia');
    assert.equal(webRequests[0].api_key,key);assert.equal(e.sourceKey.value,'');
    e.sourceMode.value='alpha';e.sourceMode.onchange();await submit();e.presetSettings.onclick();
    assert.equal(e.apiParam.value,'apikey');assert.equal(new URL(e.apiEndpoint.value).hostname,'www.alphavantage.co');
    e.sourceMode.value='custom';e.sourceMode.onchange();await submit();
    input('apiEndpoint','https://one.example.org/monthly');e.sourceKey.value=key;
    input('apiEndpoint','https://two.example.org/monthly');assert.equal(e.sourceKey.value,'');
    assert(e.keyDestination.textContent.includes('two.example.org'));
    e.apiAuth.value='none';e.apiAuth.onchange();e.apiDate.value='date';e.apiValues.value='value';
    assert(e.keyRow.hidden);await e.connectSource.onclick();
    assert.equal(webRequests[1].auth,'none');assert.equal(webRequests[1].api_key,'');
    assert(!html.includes('LOCAL ECONOMETRICS LAB · v'));
    const retailRequests=[];
    sourceFetch=async(path,opts)=>{
        const body=JSON.parse(opts.body);
        if(path==='/sources/fred'){
            retailRequests.push(body);
            return Response.json({csv:'date,MRTSSM44000USN\n2024-12-01,540000\n',filename:'FRED retail sales'});
        }
        assert.equal(path,'/analyze');
        if(!body.csv) return fetch(process.argv[2]+path,opts);
        assert(!opts.body.includes(key));
        return new Response(JSON.stringify({type:'result',report:{id:'airfare',version:1,task:'forecast',columns:[],rows:[],audit:{}}})+'\n');
    };
    e.sourceMode.value='fred';e.sourceMode.onchange();
    e.file.value='previous.csv';await e.forecast.onclick();
    assert.equal(e.file.value,'');assert.equal(e.sourceMode.value,'fred');
    assert(e.context.value.includes('MRTSSM44000USN'));assert(!e.reference.disabled);
    await e.form.onsubmit({preventDefault(){},submitter:e.reference});
    assert.equal(e.fredIds.value,'MRTSSM44000USN');assert(!e.fredIds.readOnly);
    e.sourceKey.value=key;await e.connectSource.onclick();
    assert.deepEqual(retailRequests[0],{series_ids:'MRTSSM44000USN',api_key:key,recipe:'retail_2005_2024'});
    e.sourceMode.value='local';e.sourceMode.onchange();await e.forecast.onclick();
    assert(e.context.value.includes('included U.S. monthly retail sales'));
    e.sourceMode.value='fred';e.sourceMode.onchange();await submit();
    assert.equal(e.fredIds.value,'MRTSSM44000USN');
    assert(e.sourceStatus.textContent.includes('millions of dollars through December 2024'));
    sourceFetch=(path,opts)=>fetch(process.argv[2]+path,opts);

    e['category-hybrid'].onclick(); await e.seasonForest.onclick();
    assert.equal(e.sourceMode.value,'fred');
    await e.form.onsubmit({preventDefault(){},submitter:e.reference});
    assert(!e.sourcePanel.hidden && e.fredIds.value==='MRTSSM44000USN');
    e.sourceMode.value='retail_demo';e.sourceMode.onchange();await e.seasonForest.onclick();
    assert(e.clarity.value.includes('hybrid') && !e['shelf-hybrid'].hidden);
    e['category-ml'].onclick(); assert(fields.every(f=>f.value===''));
    await e.mlLasso.onclick(); assert(e.clarity.value.includes('ML only'));
    e['category-classic'].onclick(); await e.classicAR.onclick();
    assert(e.clarity.value.includes('classical only') && e.familyLearnerLabel.hidden);
    for(const [group,card] of [['ml','buildML'],['classic','buildClassical'],['hybrid','buildHybrid']]){
        e['category-'+group].onclick(); e[card].onclick();
        assert(e.familySettings.open && !e.familyCustomize.hidden && e.analysisWorkspace.hidden);
        assert(fields.every(f=>!f.value) && e.reference.disabled);
    }
    const optionsFor=id=>[...html.split('id="'+id+'">')[1].split('</select>')[0].matchAll(/<option value="([^"]+)"/g)].map(m=>m[1]);
    assert(optionsFor('familyBase').includes('ma1') && optionsFor('familyBase').includes('ar12') && optionsFor('familyBase').includes('seasonal_naive'));
    assert(optionsFor('familyLearner').includes('boosting') && optionsFor('familyLearner').includes('elastic_net'));
    e.familySource.value='activity'; e.familySource.onchange();
    e.familyBase.value='ar3'; e.familyBase.onchange();
    e.familyLearner.value='boosting'; e.familyLearner.onchange();
    await e.applyFamily.onclick();
    assert(e.clarity.value.includes('AR(3)') && e.clarity.value.includes('Gradient boosting'));
    assert(!e.reference.disabled && e.analysisWorkspace.hidden && e.practiceNotes.hidden);
    await e.form.onsubmit({preventDefault(){},submitter:e.reference});
    assert(e.summary.textContent.includes('activity') && e.interpretation.textContent.includes('Gradient boosting'));
    e.familyBase.value='seasonal_naive'; e.familyBase.onchange();
    assert(e.reference.disabled && e.run.disabled && e.analysisWorkspace.hidden);
    console.log('Passed: custom builders, AR(3) + gradient boosting and explicit Run on simulated activity.');
    e['category-hybrid'].onclick(); await e.trendLasso.onclick();
    e.familyBase.value='ar1'; e.familyLearner.value='forest';
    await e.applyFamily.onclick(); assert(e.clarity.value.includes('AR(1)') && !e.reference.disabled);
    await e.form.onsubmit({preventDefault(){},submitter:e.reference});
    assert(e.summary.textContent.includes('Hybrid') && e.interpretation.textContent.includes('adds'));
    e['category-forecast'].onclick(); await e.familyCompare.onclick();
    await e.form.onsubmit({preventDefault(){},submitter:e.reference});
    assert(e.summary.textContent.includes('Validation recommends') && e.scores.children.length===4);
    e.familySource.value='csv'; e.familyTarget.value='sales'; e.familyUnits.value='dollars';
    await e.applyFamily.onclick(); assert(e.clarity.value.includes('sales') && e.reference.disabled);
    console.log('Passed: hybrid, ML-only, classical-only and comparison tabs, configurable pairs and CSV roles.');
    e['division-estimation'].onclick();
    assert(e['panel-forecasting'].hidden && !e['shelf-estimation-blank'].hidden);
    assert(fields.every(f=>f.value==='') && e.familyCustomize.hidden && e.reference.disabled);
    e['category-regression'].onclick();await e.estimate.onclick();
    assert.equal(e.reference.textContent,'Run regression (Python only)');assert(!e.reference.disabled);
    await e.form.onsubmit({preventDefault(){},submitter:e.reference});
    assert(e.summary.textContent.includes('adjusted coefficient') && e.summary.textContent.includes('95% HAC interval'));
    e['division-forecasting'].onclick();e['category-forecast'].onclick();
    e.sourceMode.value='retail_demo';await e.explore.onclick();assert(!e.reference.disabled);
    assert.equal(e.reference.textContent,'Run exploration (Python only)');
    await e.form.onsubmit({preventDefault(){},submitter:e.reference});
    assert(e.summary.textContent.includes('Recommend Trend + month effects + AR(1)'));
    e.exploreTarget.value='sales';e.exploreUnits.value='dollars';e.applyExplore.onclick();
    assert(e.reference.disabled);assert(e.clarity.value.includes('sales'));
    console.log('Passed: retail sales feature report, model recommendation, custom CSV prompt.');
    e['division-estimation'].onclick();e['category-ts'].onclick();
    e.sourceMode.value='local';e.sourceMode.onchange();await e.dmlExample.onclick();
    assert.equal(e.reference.textContent,'Run DML assessment (Python only)');assert(!e.reference.disabled);
    await e.form.onsubmit({preventDefault(){},submitter:e.reference});
    assert.equal(e.results.hidden,false);assert(e.summary.textContent.includes('does not pass'));
    assert.equal(e.scores.children.length,4);
    assert.equal(e.scores.children[2].children[1].textContent,'Concern');
    e.sourceMode.value='fred';e.sourceMode.onchange();
    await e.form.onsubmit({preventDefault(){},submitter:e.reference});
    assert.equal(e.fredIds.value,'CUSR0000SETG01, MJFUELUSGULF');
    await e.crossExample.onclick();assert(e.sourcePanel.hidden);assert.equal(e.dataKind.value,'cross_section');
    assert.equal(e.reference.textContent,'Run DML estimate (Python only)');
    await e.form.onsubmit({preventDefault(){},submitter:e.reference});
    assert(e.summary.textContent.includes('Coefficient on training_hours'));assert(e.interpretation.textContent.includes('800 complete independent units'));
    await e.crossAssess.onclick();await e.form.onsubmit({preventDefault(){},submitter:e.reference});
    assert.equal(e.reference.textContent,'Run DML assessment (Python only)');assert(!e.summary.textContent.includes('Coefficient on'));
    await e.tsEstimate.onclick();assert.equal(e.sourceMode.value,'ar_demo');assert(!e.reference.disabled);
    await e.form.onsubmit({preventDefault(){},submitter:e.reference});
    assert(e.summary.textContent.includes('Coefficient on activity__lag1'));
    e.sourceMode.value='fred';e.sourceMode.onchange();assert(e.context.value.includes('MJFUELUSGULF'));
    await e.form.onsubmit({preventDefault(){},submitter:e.reference});assert.equal(e.fredIds.value,'CUSR0000SETG01, MJFUELUSGULF');
    // From the exact DML-estimate panel, a changed ID bypasses the airfare transformation.
    const starter={clarity:'Estimate time-series DML on current CPIAUCSL and its first lag.',
        context:'Use monthly FRED CPIAUCSL in original units.',structure:'Screen the conditions first.',
        role:'Explain the results to an economist.',examples:'Use May CPI for June.'};
    let received=[],runsAfterImport=0;
    sourceFetch=async(path,opts)=>{
        assert(!opts.body.includes(key) || path==='/sources/fred');
        if(path==='/sources/fred'){
            const body=JSON.parse(opts.body);received.push(body);
            if(received.length===1)return Response.json({error:'Retry this import.'},{status:400});
            return Response.json({series:['CPIAUCSL'],months:120,starter,csv:'date,CPIAUCSL\n2015-01-01,234.747',filename:'FRED monthly data (CPIAUCSL)'});
        }
        const body=JSON.parse(opts.body);
        if(!body.csv) return fetch(process.argv[2]+path,opts);
        runsAfterImport++;assert.equal(body.fred_task,'ts_dml');assert.equal(body.mode,'reference');
        assert(!body.prompt.includes('fare_inflation'));assert(body.csv.includes('CPIAUCSL'));
        return new Response(JSON.stringify({type:'result',report:{id:'cpi',version:1,task:'ts_dml',columns:[],rows:[],audit:{}}})+'\n');
    };
    input('fredIds','CPIAUCSL');assert(e.sourceStatus.textContent.includes('original'));
    input('fredIds',' mjfuelusgulf , cusr0000setg01 ');assert(e.sourceStatus.textContent.includes('percentage changes'));
    input('fredIds','`CPIAUCSL`');assert.equal(e.connectSource.textContent,'Import and review prompt');
    e.sourceKey.value=key;await e.connectSource.onclick();
    assert.equal(e.fredIds.value,'`CPIAUCSL`');assert.equal(e.sourceKey.value,'');assert(!e.sourcePanel.hidden);
    e.sourceKey.value=key;await e.connectSource.onclick();
    assert.deepEqual(received.at(-1),{series_ids:'CPIAUCSL',api_key:key,starter_task:'ts_dml'});
    assert.equal(runsAfterImport,0);assert(e.sourcePanel.hidden);assert(!e.changeFred.hidden);
    assert.deepEqual(fields.map(f=>f.value),fields.map(f=>starter[f.id]));assert(!e.reference.disabled);
    await e.form.onsubmit({preventDefault(){},submitter:e.reference});assert.equal(runsAfterImport,1);
    e.changeFred.onclick();assert.equal(e.fredIds.value,'`CPIAUCSL`');assert(!e.sourcePanel.hidden);
    e.sourceMode.value='local';e.sourceMode.onchange();assert(e.changeFred.hidden);
    // Choosing a worked card only prepares its inputs. Every card must run on explicit submission.
    const workedCards = [
        ['forecasting','hybrid',['seasonForest','trendLasso','arForest'],'family_forecast'],
        ['forecasting','ml',['mlForest','mlLasso'],'family_forecast'],
        ['forecasting','ml',['forecast'],'forecast'],
        ['forecasting','classic',['classicAR','classicMA','classicTrend'],'family_forecast'],
        ['forecasting','forecast',['familyCompare'],'family_forecast'],
        ['forecasting','forecast',['explore'],'exploration'],
        ['estimation','regression',['estimate'],'estimation'],
        ['estimation','cross',['crossAssess','crossExample'],'cross_dml'],
        ['estimation','ts',['dmlExample'],'rcf_dml_screen'],
        ['estimation','ts',['tsEstimate'],'ts_dml']
    ];
    let attempts, completed;
    sourceFetch=async (path,opts)=>{
        assert.equal(path,'/analyze');
        assert.equal(JSON.parse(opts.body).mode,'reference');
        attempts++;
        const res=await fetch(process.argv[2]+path,opts);
        completed=JSON.parse((await res.clone().text()).trim().split('\n').at(-1));
        return res;
    };
    for(const [division,group,cards,task] of workedCards){
        e['division-'+division].onclick();
        e['category-'+group].onclick();
        for(const card of cards){
            // Start with a different included source to catch data carried over from another example.
            e.sourceMode.value=division==='forecasting'?'ar_demo':'retail_demo';
            e.sourceMode.onchange();
            attempts=0;completed=null;
            await e[card].onclick();
            assert.equal(attempts,0,card);
            assert(fields.every(f=>f.value.trim()),card);
            assert(e.analysisWorkspace.hidden && e.results.hidden && e.practiceNotes.hidden,card);
            assert.equal(e.trace.children.length,0,card);
            assert.equal(e.summary.textContent,'',card);
            assert.equal(e.versions.children.length,0,card);
            assert(!e.reference.disabled && !e.reference.hidden,card);
            await e.form.onsubmit({preventDefault(){},submitter:e.reference});
            assert.equal(attempts,1,card);
            assert(!e.analysisWorkspace.hidden,card);
            assert.equal(completed?.type,'result',card+' '+JSON.stringify(completed));
            const rep=completed.report;
            assert.equal(rep.task,task,card);
            assert.equal(rep.audit.agent_mode,'Python reference only',card);
            assert.equal(rep.source, division==='forecasting'?'Included U.S. retail sales, Jan 2005–Dec 2024':
                task==='ts_dml'?'Bundled simulated AR activity':task==='cross_dml'?'Bundled simulated independent households':'Bundled simulated monthly data',card);
            assert(rep.rows.length && !e.results.hidden && !e.summary.textContent.includes('undefined'),card);
            if(card==='classicMA'){
                assert.equal(rep.audit.base,'ma1');
                assert(e.interpretation.textContent.includes('last estimated shock'));
            }
            if(card==='dmlExample'){
                assert.equal(rep.audit.target,'fare_inflation');
                assert.equal(rep.audit.term,'fuel_inflation__lag1');
                assert.equal(rep.rows.length,3);
            }
        }
    }
    input('role','Explain the estimate for another audience.');
    assert(e.analysisWorkspace.hidden && e.results.hidden && e.applyFeedback.disabled);
    assert.equal(e.versions.children.length,0);
    console.log('Passed: all 16 worked cards waited for Run and then returned results using the matching data.');
    console.log('Passed: edited DML IDs, preset restoration, retry, raw import, prompt review and custom Python run.');
    console.log('Passed: independent-unit cross-sectional assessment and estimate, monthly AR estimation and FRED handoff.');
    console.log('Passed: DML example, three evidence rows, overall verdict and FRED reference form.');
    console.log('Passed: Forecast example in either click order, editable preset IDs, no stale upload, FRED reference and version-free heading.');
    console.log('Passed: provider detection, EIA and Alpha Vantage presets, custom host change and no-key import.');
    console.log('Passed: masked FRED form, safe retry, fetched data reused on clarification, key excluded from every analysis request.');
    console.log('Passed: instant blank highlights, automatic choices, current draft, explicit opt-in, preserved typing and focus, stale requests, manual entry and toggle.');
}
test().catch(error=>{console.error(error);process.exitCode=1});
