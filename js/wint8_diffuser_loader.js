import { app } from "../../scripts/app.js";

const NODE_TYPE = "WINT8DiffuserLoader";

// ── Sparkle system ────────────────────────────────────────────────────────────
class SparkleSystem {
    constructor(maxParticles = 12) { this.particles = []; this.max = maxParticles; }
    _spawn(w, h, yOff) {
        const perim = 2*(w+h); let d = Math.random()*perim, x, y;
        if      (d < w)          { x = d;              y = yOff; }
        else if (d < w+h)        { x = w;               y = yOff+(d-w); }
        else if (d < 2*w+h)      { x = w-(d-w-h);       y = yOff+h; }
        else                     { x = 0;               y = yOff+h-(d-2*w-h); }
        this.particles.push({ x, y,
            vx:(Math.random()-0.5)*0.6, vy:(Math.random()-0.5)*0.6,
            life:1.0, decay:0.008+Math.random()*0.012, size:1.2+Math.random()*2.0 });
    }
    update(w, h, yOff) {
        while (this.particles.length < this.max) this._spawn(w, h, yOff);
        for (let i = this.particles.length-1; i >= 0; i--) {
            const p = this.particles[i];
            p.x += p.vx; p.y += p.vy; p.life -= p.decay;
            if (p.life <= 0) this.particles.splice(i, 1);
        }
    }
    draw(ctx) {
        for (const p of this.particles) {
            ctx.save();
            ctx.globalAlpha = p.life*0.9; ctx.shadowColor = "#ffcc66";
            ctx.shadowBlur = 6+p.size*2; ctx.fillStyle = "#ffe0a0";
            const s = p.size;
            ctx.beginPath(); ctx.moveTo(p.x,p.y-s); ctx.lineTo(p.x+s*0.3,p.y);
            ctx.lineTo(p.x,p.y+s); ctx.lineTo(p.x-s*0.3,p.y); ctx.closePath(); ctx.fill();
            ctx.beginPath(); ctx.moveTo(p.x-s,p.y); ctx.lineTo(p.x,p.y+s*0.3);
            ctx.lineTo(p.x+s,p.y); ctx.lineTo(p.x,p.y-s*0.3); ctx.closePath(); ctx.fill();
            ctx.restore();
        }
    }
}

// ── Amber breathing glow ──────────────────────────────────────────────────────
function drawAmberGlow(ctx, node, sparkles) {
    if (node.flags?.collapsed) return;
    const w = node.size[0], h = node.size[1] + LiteGraph.NODE_TITLE_HEIGHT;
    const yOff = -LiteGraph.NODE_TITLE_HEIGHT, r = 8;
    const t = Date.now()/1000;
    const pulse  = 0.5+0.5*Math.sin(t*(2*Math.PI/3));
    const pulse2 = 0.5+0.5*Math.sin(t*(2*Math.PI/5)+1.0);
    app.graph.setDirtyCanvas(true, false);
    ctx.save();
    ctx.shadowColor="#cc8800"; ctx.shadowBlur=28+pulse*30; ctx.strokeStyle="#cc8800";
    ctx.lineWidth=1; ctx.globalAlpha=0.12+pulse*0.15;
    ctx.beginPath(); ctx.roundRect(-2,yOff-2,w+4,h+4,r+2); ctx.stroke();
    ctx.shadowColor="#f5a623"; ctx.shadowBlur=18+pulse*22; ctx.strokeStyle="#f5a623";
    ctx.lineWidth=2; ctx.globalAlpha=0.30+pulse*0.40;
    ctx.beginPath(); ctx.roundRect(0,yOff,w,h,r); ctx.stroke();
    ctx.shadowBlur=8+pulse2*10; ctx.globalAlpha=0.55+pulse2*0.35;
    ctx.lineWidth=1.5; ctx.strokeStyle="#ffd080";
    ctx.beginPath(); ctx.roundRect(1,yOff+1,w-2,h-2,r); ctx.stroke();
    ctx.shadowColor="#ffe0a0"; ctx.shadowBlur=8;
    ctx.globalAlpha=0.3+pulse*0.5; ctx.fillStyle="#ffe0a0";
    const dotR = 2+pulse*1.5;
    for (const [cx,cy] of [[2,yOff+2],[w-2,yOff+2],[2,yOff+h-2],[w-2,yOff+h-2]]) {
        ctx.beginPath(); ctx.arc(cx,cy,dotR,0,Math.PI*2); ctx.fill();
    }
    ctx.restore();
    sparkles.update(w,h,yOff); sparkles.draw(ctx);
}

// ── Extension ─────────────────────────────────────────────────────────────────
app.registerExtension({
    name: "WINT8.DiffuserLoader",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origCreated?.call(this);
            this.color     = "#2a1a00";
            this.bgcolor   = "#1a1000";
            this._sparkles = new SparkleSystem(12);
            this.title     = "🔥 WINT8 Diffuser Loader";
        };

        const origBg = nodeType.prototype.onDrawBackground;
        nodeType.prototype.onDrawBackground = function (ctx) {
            origBg?.call(this, ctx);
            if (!this._sparkles) this._sparkles = new SparkleSystem(12);
            drawAmberGlow(ctx, this, this._sparkles);
        };

        const origFg = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            origFg?.call(this, ctx);
            if (this.flags?.collapsed) return;
            const W = this.size[0];
            ctx.save();
            // Badge
            ctx.font="bold 10px sans-serif"; ctx.textAlign="right";
            ctx.textBaseline="alphabetic"; ctx.fillStyle="#f5a623";
            ctx.shadowColor="#f5a623"; ctx.shadowBlur=6;
            ctx.fillText("⚡ WINT8", W-8, 14);
            // INT8 pill
            ctx.shadowBlur=0; ctx.shadowColor="transparent";
            const label = "INT8";
            ctx.font="bold 9px monospace";
            const tw = ctx.measureText(label).width;
            const pad=5, pw=tw+pad*2, ph=14;
            const px=8, py=2;
            ctx.beginPath(); ctx.roundRect(px,py,pw,ph,4);
            ctx.fillStyle="#2a1800"; ctx.fill();
            ctx.strokeStyle="#cc8800"; ctx.lineWidth=1; ctx.stroke();
            ctx.fillStyle="#ffcc66"; ctx.textAlign="center"; ctx.textBaseline="middle";
            ctx.fillText(label, px+pw/2, py+ph/2);
            ctx.restore();
        };
    },
});
