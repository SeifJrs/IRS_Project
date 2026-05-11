// ═══ PARTICLE BACKGROUND ═══
(function(){
    const c=document.getElementById('bg-canvas'),ctx=c.getContext('2d');
    let w,h,particles=[];
    function resize(){w=c.width=window.innerWidth;h=c.height=window.innerHeight;}
    window.addEventListener('resize',resize);resize();
    for(let i=0;i<60;i++) particles.push({x:Math.random()*w,y:Math.random()*h,r:Math.random()*2+0.5,dx:(Math.random()-0.5)*0.3,dy:(Math.random()-0.5)*0.3,o:Math.random()*0.3+0.05});
    function draw(){
        ctx.clearRect(0,0,w,h);
        particles.forEach(p=>{
            p.x+=p.dx;p.y+=p.dy;
            if(p.x<0)p.x=w;if(p.x>w)p.x=0;if(p.y<0)p.y=h;if(p.y>h)p.y=0;
            ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
            ctx.fillStyle=`rgba(108,99,255,${p.o})`;ctx.fill();
        });
        // Draw connections
        for(let i=0;i<particles.length;i++){
            for(let j=i+1;j<particles.length;j++){
                const dx=particles[i].x-particles[j].x,dy=particles[i].y-particles[j].y;
                const dist=Math.sqrt(dx*dx+dy*dy);
                if(dist<120){
                    ctx.beginPath();ctx.moveTo(particles[i].x,particles[i].y);
                    ctx.lineTo(particles[j].x,particles[j].y);
                    ctx.strokeStyle=`rgba(108,99,255,${0.06*(1-dist/120)})`;
                    ctx.lineWidth=0.5;ctx.stroke();
                }
            }
        }
        requestAnimationFrame(draw);
    }
    draw();
})();

// ═══ ANIMATED COUNTERS ═══
function animateCounters(){
    document.querySelectorAll('.stat-value[data-count]').forEach(el=>{
        const target=parseInt(el.dataset.count);
        if(isNaN(target))return;
        let current=0;const step=Math.max(1,Math.floor(target/60));
        const timer=setInterval(()=>{
            current+=step;
            if(current>=target){current=target;clearInterval(timer);}
            el.textContent=current.toLocaleString();
        },20);
    });
}

// ═══ TOAST ═══
function showToast(msg,icon='✅'){
    const t=document.createElement('div');t.className='toast';
    t.innerHTML=`<span>${icon}</span> ${msg}`;
    document.body.appendChild(t);
    setTimeout(()=>t.remove(),3000);
}

// ═══ NAVIGATION ═══
function showPage(pageId){
    document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active'));
    document.getElementById('page-'+pageId).classList.add('active');
    document.getElementById('nav-'+pageId).classList.add('active');
}

function switchTab(btn,tabId){
    const bar=btn.parentElement;
    bar.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    bar.parentElement.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
}

// ═══ LOADING ═══
function showLoading(text){
    document.getElementById('loadingText').textContent=text||'Loading...';
    document.getElementById('loadingOverlay').classList.add('active');
}
function hideLoading(){document.getElementById('loadingOverlay').classList.remove('active');}

// ═══ USER INFO ═══
function onUserChange(){
    const uid=document.getElementById('userSelect').value;
    fetch('/api/user_info?user_id='+uid).then(r=>r.json()).then(data=>{
        document.getElementById('userStats').innerHTML='📈 <span>'+data.n_ratings+'</span> ratings · ❤️ '+data.prefs;
    });
}

// ═══ PRODUCT CARD HTML ═══
function productCardHTML(items){
    if(!items||items.length===0) return '<div class="analysis-box">⚠️ No recommendations found for this user/configuration.</div>';
    let html='<div class="product-grid">';
    items.forEach((item,idx)=>{
        const stars='⭐'.repeat(Math.min(5,Math.round(item.predicted_rating||0)));
        html+=`<div class="product-card" style="animation-delay:${idx*0.06}s">
            <div class="product-header">
                <div class="product-name">${item.product_name}</div>
                <div class="rating-badge">${(item.predicted_rating||0).toFixed(1)} ${stars}</div>
            </div>
            <div class="product-meta">
                <span class="meta-tag"><i class="fas fa-tag"></i> ${item.category}</span>
                <span class="meta-tag"><i class="fas fa-building"></i> ${item.brand}</span>
                <span class="meta-tag"><i class="fas fa-dollar-sign"></i> ${item.price.toFixed(2)}</span>
            </div>
            <div class="product-explanation">${item.explanation||''}</div>
        </div>`;
    });
    return html+'</div>';
}

// ═══ CF ═══
function getCFRecommendations(){
    const uid=document.getElementById('userSelect').value;
    const method=document.getElementById('cfMethod').value;
    const topN=document.getElementById('topN').value;
    showLoading('Computing CF recommendations...');
    fetch(`/api/recommend/cf?user_id=${uid}&method=${method}&top_n=${topN}`).then(r=>r.json()).then(data=>{
        hideLoading();
        const names={'user_cosine':'User-Based CF (Cosine)','user_pearson':'User-Based CF (Pearson)','item_cosine':'Item-Based CF (Cosine)','svd':'Matrix Factorization (SVD)'};
        document.getElementById('cfResults').innerHTML='<h2 class="section-title"><i class="fas fa-gift"></i> Top '+topN+' — '+names[method]+'</h2>'+productCardHTML(data.recommendations);
        showToast('Recommendations generated!','🎯');
    });
}

function compareCFMethods(){
    const uid=document.getElementById('userSelect').value;
    showLoading('Comparing all 4 CF methods...');
    fetch(`/api/compare_cf?user_id=${uid}`).then(r=>r.json()).then(data=>{
        hideLoading();
        let html='<h2 class="section-title"><i class="fas fa-columns"></i> All CF Methods Comparison</h2><div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">';
        for(const[method,items]of Object.entries(data)){
            html+='<div class="chart-container"><h3 style="margin-bottom:0.8rem;font-size:0.95rem;font-weight:700;">'+method+'</h3>';
            html+='<table class="data-table"><tr><th>Product</th><th>Category</th><th>Rating</th></tr>';
            items.forEach(item=>{html+=`<tr><td>${item.product_name}</td><td>${item.category}</td><td>${item.predicted_rating.toFixed(2)}</td></tr>`;});
            html+='</table></div>';
        }
        html+='</div>';
        document.getElementById('cfComparison').innerHTML=html;
        showToast('All methods compared!','📊');
    });
}

// ═══ CB ═══
function getCBRecommendations(){
    const uid=document.getElementById('userSelect').value;
    const topN=document.getElementById('topN').value;
    showLoading('Analyzing content similarity...');
    fetch(`/api/recommend/cb?user_id=${uid}&top_n=${topN}`).then(r=>r.json()).then(data=>{
        hideLoading();
        document.getElementById('cbResults').innerHTML='<h2 class="section-title"><i class="fas fa-gift"></i> Top '+topN+' Content-Based Recommendations</h2>'+productCardHTML(data.recommendations);
        showToast('Content analysis complete!','📝');
    });
}

// ═══ KB ═══
function getKBRecommendations(){
    const uid=document.getElementById('userSelect').value;
    const topN=document.getElementById('topN').value;
    const cat=document.getElementById('kbCategory').value;
    const brand=document.getElementById('kbBrand').value;
    const minP=document.getElementById('kbMinPrice').value;
    const maxP=document.getElementById('kbMaxPrice').value;
    const minR=document.getElementById('kbMinRating').value;
    showLoading('Filtering products...');
    let url=`/api/recommend/kb?user_id=${uid}&top_n=${topN}`;
    if(cat)url+=`&category=${encodeURIComponent(cat)}`;
    if(brand)url+=`&brand=${encodeURIComponent(brand)}`;
    if(minP)url+=`&min_price=${minP}`;if(maxP)url+=`&max_price=${maxP}`;
    if(minR)url+=`&min_rating=${minR}`;
    fetch(url).then(r=>r.json()).then(data=>{
        hideLoading();
        document.getElementById('kbResults').innerHTML='<h2 class="section-title"><i class="fas fa-gift"></i> Top '+topN+' Knowledge-Based Recommendations</h2>'+productCardHTML(data.recommendations);
        showToast('Filters applied!','🎯');
    });
}

// ═══ EVALUATION ═══
function runEvaluation(){
    const k=document.getElementById('evalK').value;
    const btn=document.getElementById('evalBtn');
    btn.disabled=true;
    showLoading('Running full evaluation — this may take a minute...');
    fetch(`/api/evaluate?k=${k}`).then(r=>r.json()).then(data=>{
        hideLoading();btn.disabled=false;
        renderEvaluation(data);
        showToast('Evaluation complete!','🚀');
    }).catch(e=>{
        hideLoading();btn.disabled=false;
        document.getElementById('evalResults').innerHTML='<div class="analysis-box">❌ Error: '+e.message+'</div>';
    });
}

function renderEvaluation(data){
    let html='';
    // Best metric cards
    html+='<div class="stats-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:1.5rem;">';
    const metrics=['Precision@K','Recall@K','NDCG@K','RMSE'];
    const icons=['🎯','📈','📊','📉'];
    metrics.forEach((m,i)=>{
        let best,bestMethod;
        if(m==='RMSE'){
            const valid=data.results.filter(r=>r[m]!==null&&!isNaN(r[m]));
            if(valid.length){best=Math.min(...valid.map(r=>r[m]));bestMethod=valid.find(r=>r[m]===best).Method;}
        }else{
            best=Math.max(...data.results.map(r=>r[m]||0));
            bestMethod=data.results.find(r=>(r[m]||0)===best).Method;
        }
        html+=`<div class="stat-card"><div style="font-size:0.68rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px;">Best ${m}</div>
        <div class="stat-value" style="font-size:1.6rem;">${best!==undefined?best.toFixed(4):'N/A'}</div>
        <div style="font-size:0.78rem;color:var(--text-secondary);margin-top:0.2rem;">${bestMethod||''}</div></div>`;
    });
    html+='</div>';

    // Table
    html+='<h2 class="section-title"><i class="fas fa-table"></i> Evaluation Results</h2><div class="chart-container"><table class="data-table">';
    html+='<tr><th>Method</th><th>Precision@K</th><th>Recall@K</th><th>NDCG@K</th><th>RMSE</th><th>Users</th></tr>';
    data.results.forEach(row=>{
        html+=`<tr><td><strong>${row.Method}</strong></td>
        <td>${row['Precision@K']!==null?row['Precision@K'].toFixed(4):'N/A'}</td>
        <td>${row['Recall@K']!==null?row['Recall@K'].toFixed(4):'N/A'}</td>
        <td>${row['NDCG@K']!==null?row['NDCG@K'].toFixed(4):'N/A'}</td>
        <td>${row['RMSE']!==null&&!isNaN(row['RMSE'])?row['RMSE'].toFixed(4):'N/A'}</td>
        <td>${row.num_users_evaluated||'N/A'}</td></tr>`;
    });
    html+='</table></div>';

    // Charts
    html+='<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">';
    html+='<div class="chart-container"><canvas id="evalBarChart" height="300"></canvas></div>';
    html+='<div class="chart-container"><canvas id="evalRmseChart" height="300"></canvas></div></div>';

    // Analysis
    html+='<h2 class="section-title" style="margin-top:1.5rem;"><i class="fas fa-microscope"></i> Detailed Analysis</h2>';
    html+='<div class="analysis-box">'+data.analysis.replace(/\\n/g,'<br>').replace(/\n/g,'<br>').replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>')+'</div>';

    // Comparison table
    html+=`<h2 class="section-title" style="margin-top:1.5rem;"><i class="fas fa-balance-scale"></i> Approach Comparison</h2>
    <div class="chart-container"><table class="comparison-table">
    <tr><th>Aspect</th><th>Collaborative Filtering</th><th>Content-Based</th><th>Knowledge-Based</th></tr>
    <tr><td><strong>Cold Start</strong></td><td>❌ Needs ratings</td><td>✅ Uses item features</td><td>✅ Uses constraints</td></tr>
    <tr><td><strong>Diversity</strong></td><td>✅ High</td><td>⚠️ Filter bubble risk</td><td>⚠️ Limited by rules</td></tr>
    <tr><td><strong>Explainability</strong></td><td>⚠️ Moderate</td><td>✅ Feature-based</td><td>✅ Constraint-based</td></tr>
    <tr><td><strong>Scalability</strong></td><td>⚠️ Matrix size</td><td>✅ Good</td><td>✅ Good</td></tr>
    <tr><td><strong>Serendipity</strong></td><td>✅ High</td><td>❌ Low</td><td>❌ Low</td></tr>
    <tr><td><strong>Best For</strong></td><td>Dense rating data</td><td>Rich item features</td><td>New users / explicit needs</td></tr>
    </table></div>`;

    document.getElementById('evalResults').innerHTML=html;

    // Draw charts
    setTimeout(()=>{
        const methods=data.results.map(r=>r.Method);
        const prec=data.results.map(r=>r['Precision@K']||0);
        const rec=data.results.map(r=>r['Recall@K']||0);
        const ndcg=data.results.map(r=>r['NDCG@K']||0);
        const rmse=data.results.map(r=>r['RMSE']||0);
        const chartOpts={responsive:true,plugins:{legend:{labels:{color:'#a0a0c8',font:{family:'Inter'}}},title:{color:'#f0f0ff',font:{family:'Inter',weight:700}}},scales:{x:{ticks:{color:'#a0a0c8',maxRotation:25,font:{family:'Inter'}},grid:{color:'rgba(255,255,255,0.04)'}},y:{ticks:{color:'#a0a0c8',font:{family:'Inter'}},grid:{color:'rgba(255,255,255,0.04)'}}}};

        new Chart(document.getElementById('evalBarChart'),{type:'bar',data:{labels:methods,datasets:[
            {label:'Precision@K',data:prec,backgroundColor:'rgba(108,99,255,0.7)',borderRadius:6},
            {label:'Recall@K',data:rec,backgroundColor:'rgba(16,185,129,0.7)',borderRadius:6},
            {label:'NDCG@K',data:ndcg,backgroundColor:'rgba(245,158,11,0.7)',borderRadius:6}
        ]},options:{...chartOpts,plugins:{...chartOpts.plugins,title:{display:true,text:'Method Comparison — Higher is Better',...chartOpts.plugins.title}}}});

        new Chart(document.getElementById('evalRmseChart'),{type:'bar',data:{labels:methods,datasets:[
            {label:'RMSE',data:rmse,backgroundColor:'rgba(239,68,68,0.7)',borderRadius:6}
        ]},options:{...chartOpts,plugins:{...chartOpts.plugins,title:{display:true,text:'RMSE Comparison — Lower is Better',...chartOpts.plugins.title}}}});
    },200);
}

// ═══ HOME CHARTS ═══
function initHomeCharts(catData, ratingData){
    const baseOpts={responsive:true,plugins:{legend:{display:false},title:{color:'#f0f0ff',font:{family:'Inter',weight:700,size:14}}},scales:{x:{ticks:{color:'#a0a0c8',font:{family:'Inter'}},grid:{color:'rgba(255,255,255,0.04)'}},y:{ticks:{color:'#a0a0c8',font:{family:'Inter'}},grid:{color:'rgba(255,255,255,0.04)'}}}};

    new Chart(document.getElementById('catChart'),{type:'bar',data:{labels:catData.labels,datasets:[{
        label:'Products',data:catData.values,borderRadius:8,
        backgroundColor:['rgba(108,99,255,0.7)','rgba(168,85,247,0.7)','rgba(236,72,153,0.7)','rgba(16,185,129,0.7)','rgba(245,158,11,0.7)','rgba(239,68,68,0.7)','rgba(6,182,212,0.7)','rgba(139,92,246,0.7)']
    }]},options:{...baseOpts,plugins:{...baseOpts.plugins,title:{display:true,text:'Products per Category',...baseOpts.plugins.title}}}});

    new Chart(document.getElementById('ratingChart'),{type:'bar',data:{labels:ratingData.labels,datasets:[{
        label:'Count',data:ratingData.values,borderRadius:8,
        backgroundColor:'rgba(108,99,255,0.7)',hoverBackgroundColor:'rgba(108,99,255,0.9)'
    }]},options:{...baseOpts,plugins:{...baseOpts.plugins,title:{display:true,text:'Rating Distribution',...baseOpts.plugins.title}}}});
}
