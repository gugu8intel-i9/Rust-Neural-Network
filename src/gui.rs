//! Interactive GUI: self-contained HTML dashboard generator.
//!
//! Generates zero-dependency HTML files that open in any browser.

use std::fmt::Write;

/// Model architecture visualization builder (SVG flow diagram).
#[derive(Debug, Clone)]
pub struct ModelDashboard {
    title: String,
    layers: Vec<(String, usize, usize)>,
}

impl ModelDashboard {
    pub fn new(title: impl Into<String>) -> Self {
        ModelDashboard { title: title.into(), layers: Vec::new() }
    }
    pub fn layer(mut self, name: &str, inp: usize, out: usize, _init: &str) -> Self {
        self.layers.push((name.into(), inp, out));
        self
    }
    pub fn render(&self) -> String {
        let tp: usize = self.layers.iter().map(|(n,i,o)| if n.starts_with("Linear") { i*o+o } else {0}).sum();
        let bw=160; let bh=60; let gap=50;
        let sw = self.layers.len()*(bw+gap)+gap;
        let sh = 200;
        let mut svg = String::new();
        for (i,(name,inp,out)) in self.layers.iter().enumerate() {
            let x = gap+i*(bw+gap); let y=(sh-bh)/2;
            let col = if name.starts_with("Linear") {"#4A90D9"} else if name.contains("ReLU") {"#D94A4A"} else if name.contains("Norm") {"#2ECC71"} else {"#95A5A6"};
            if i>0 { let px=gap+(i-1)*(bw+gap)+bw; let _=write!(svg,"<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#555\" stroke-width=\"2\"/>",px,y+bh/2,x,y+bh/2);}
            let _=write!(svg,"<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" rx=\"8\" fill=\"{}\" stroke=\"#333\"/>",x,y,bw,bh,col);
            let _=write!(svg,"<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" fill=\"white\" font-size=\"14\">{}</text>",x+bw/2,y+22,name);
            if *inp>0 { let _=write!(svg,"<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" fill=\"#ddd\" font-size=\"11\">{} to {}</text>",x+bw/2,y+42,inp,out);}
        }
        let kb = tp as f64 * 4.0/1024.0;
        let mut h = String::new();
        h.push_str("<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>");
        h.push_str(&self.title);
        h.push_str("</title><style>body{font-family:sans-serif;background:#1a1a2e;color:#eee;margin:0;padding:20px}h1{color:#e94560}.stats{display:flex;gap:20px;margin:20px 0}.stat{background:#16213e;border-radius:8px;padding:12px 20px}.stat-value{font-size:24px;font-weight:bold;color:#e94560}.stat-label{font-size:12px;color:#888}svg{max-width:100%}</style></head><body><h1>");
        h.push_str(&self.title);
        h.push_str("</h1><div class=\"stats\"><div class=\"stat\"><div class=\"stat-value\">");
        let _ = write!(h,"{}</div><div class=\"stat-label\">Layers</div></div>",self.layers.len());
        let _ = write!(h,"<div class=\"stat\"><div class=\"stat-value\">{}</div><div class=\"stat-label\">Parameters</div></div>",tp);
        let _ = write!(h,"<div class=\"stat\"><div class=\"stat-value\">{:.1} KB</div><div class=\"stat-label\">Size</div></div>",kb);
        let _ = write!(h,"</div><svg viewBox=\"0 0 {} {}\" xmlns=\"http://www.w3.org/2000/svg\">{}</svg></body></html>",sw,sh,svg);
        h
    }
}

/// Training metrics dashboard builder (SVG line chart).
#[derive(Debug, Clone)]
pub struct TrainingDashboard {
    title: String,
    points: Vec<(usize, f64)>,
}

impl TrainingDashboard {
    pub fn new(title: impl Into<String>) -> Self {
        TrainingDashboard { title: title.into(), points: Vec::new() }
    }
    pub fn add_point(&mut self, epoch: usize, loss: f64) { self.points.push((epoch, loss)); }
    pub fn render(&self) -> String {
        let p=&self.points;
        if p.is_empty() { return format!("<html><body><h1>{}</h1><p>No data.</p></body></html>",self.title); }
        let me=p.iter().map(|(e,_)|*e).max().unwrap_or(1);
        let ml=p.iter().map(|(_,l)|*l).fold(0.0f64,f64::max).max(0.01);
        let mn=p.iter().map(|(_,l)|*l).fold(f64::INFINITY,f64::min);
        let r=(ml-mn).max(0.001);
        let path:String=p.iter().map(|(e,l)|{let x=40.0+(*e as f64/me as f64)*520.0;let y=40.0+(1.0-(l-mn)/r)*220.0;format!("{:.1},{:.1}",x,y)}).collect::<Vec<_>>().join(" ");
        let fl=p.last().map(|(_,l)|*l).unwrap_or(0.0);
        let il=p.first().map(|(_,l)|*l).unwrap_or(0.0);
        let imp=if il>0.0{(1.0-fl/il)*100.0}else{0.0};
        let mut h=String::new();
        h.push_str("<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Training</title><style>body{font-family:sans-serif;background:#0f0f23;color:#eee;margin:0;padding:20px}h1{color:#e94560}.stats{display:flex;gap:16px;margin:16px 0}.stat{background:#1a1a2e;border-radius:8px;padding:10px 18px;text-align:center}.sv{font-size:22px;font-weight:bold;color:#e94560}.sl{font-size:11px;color:#666;text-transform:uppercase}svg{background:#16213e;border-radius:12px}</style></head><body><h1>Training: ");
        h.push_str(&self.title);
        let _=write!(h,"</h1><p>{} epochs</p><div class=\"stats\">",p.len());
        let _=write!(h,"<div class=\"stat\"><div class=\"sv\">{:.4}</div><div class=\"sl\">Initial</div></div>",il);
        let _=write!(h,"<div class=\"stat\"><div class=\"sv\">{:.4}</div><div class=\"sl\">Final</div></div>",fl);
        let _=write!(h,"<div class=\"stat\"><div class=\"sv\">{:.1}%</div><div class=\"sl\">Improvement</div></div></div>",imp);
        let _=write!(h,"<svg viewBox=\"0 0 600 300\" xmlns=\"http://www.w3.org/2000/svg\">");
        let _=write!(h,"<line x1=\"40\" y1=\"40\" x2=\"40\" y2=\"260\" stroke=\"#333\"/><line x1=\"40\" y1=\"260\" x2=\"560\" y2=\"260\" stroke=\"#333\"/>");
        let _=write!(h,"<polyline points=\"{}\" fill=\"none\" stroke=\"#e94560\" stroke-width=\"2.5\"/>",path);
        let _=write!(h,"<text x=\"30\" y=\"35\" fill=\"#888\" font-size=\"11\">{:.3}</text>",ml);
        let _=write!(h,"<text x=\"30\" y=\"275\" fill=\"#888\" font-size=\"11\">{:.3}</text>",mn);
        let _=write!(h,"<text x=\"540\" y=\"278\" fill=\"#888\" font-size=\"11\">Epoch {}</text>",me);
        h.push_str("</svg></body></html>");
        h
    }
}

/// Tensor heatmap HTML.
pub fn tensor_heatmap_html(tensor: &crate::tensor::Tensor, max_display: usize) -> String {
    let data:Vec<f32>=tensor.data().iter().copied().take(max_display).collect();
    let shape=tensor.shape();
    let cols=shape.last().copied().unwrap_or(data.len()).max(1);
    let mx=data.iter().copied().fold(f32::NEG_INFINITY,f32::max);
    let mn=data.iter().copied().fold(f32::INFINITY,f32::min);
    let r=(mx-mn).max(1e-8);
    let mut cells=String::new();
    for (i,&v) in data.iter().enumerate() {
        let norm=((v-mn)/r).clamp(0.0,1.0);
        let hue=(1.0-norm)*240.0;
        let _=write!(cells,"<td style=\"background:hsl({:.0},70%,50%);color:#fff;font-size:9px;text-align:center;padding:2px 4px\">{:.2}</td>",hue,v);
        if (i+1)%cols==0 && i+1<data.len() { cells.push_str("</tr><tr>"); }
    }
    format!("<table style=\"border-collapse:collapse;font-family:monospace\"><tr>{}</tr></table>",cells)
}

/// Write HTML to file and open in browser.
pub fn launch(html: &str, path: &str) -> std::io::Result<()> {
    std::fs::write(path, html)?;
    #[cfg(target_os="linux")]{std::process::Command::new("xdg-open").arg(path).spawn().ok();}
    #[cfg(target_os="macos")]{std::process::Command::new("open").arg(path).spawn().ok();}
    #[cfg(target_os="windows")]{std::process::Command::new("cmd").args(["/C","start",path]).spawn().ok();}
    Ok(())
}

/// Combined dashboard.
pub fn full_dashboard(title: &str, model: &ModelDashboard, training: &TrainingDashboard) -> String {
    let mut h=String::new();
    h.push_str("<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>");
    h.push_str(title);
    h.push_str("</title><style>body{font-family:sans-serif;background:#0f0f23;color:#eee;margin:0;padding:20px}h1,h2{color:#e94560}.sec{background:#16213e;border-radius:12px;padding:20px;margin:16px 0}</style></head><body><h1>");
    h.push_str(title);
    h.push_str("</h1><div class=\"sec\"><h2>Model Architecture</h2>");
    h.push_str(&model.render());
    h.push_str("</div><div class=\"sec\"><h2>Training Metrics</h2>");
    h.push_str(&training.render());
    h.push_str("</div></body></html>");
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_html() {
        let d=ModelDashboard::new("MLP").layer("Linear",10,20,"he").layer("ReLU",20,20,"").layer("Linear",20,5,"he");
        let h=d.render();
        assert!(h.contains("<html")&&h.contains("Linear")&&h.contains("Parameters"));
    }
    #[test]
    fn training_html() {
        let mut d=TrainingDashboard::new("Run");
        d.add_point(1,3.0); d.add_point(2,1.0);
        let h=d.render();
        assert!(h.contains("<svg")&&h.contains("polyline"));
    }
    #[test]
    fn training_empty() {
        let d=TrainingDashboard::new("E");
        assert!(d.render().contains("No data"));
    }
    #[test]
    fn heatmap() {
        let t=crate::tensor::Tensor::from_vec(vec![1.0,2.0,3.0,4.0],vec![2,2]);
        let h=tensor_heatmap_html(&t,100);
        assert!(h.contains("<td")&&h.contains("hsl("));
    }
    #[test]
    fn full_dash() {
        let m=ModelDashboard::new("M").layer("Linear",4,8,"he");
        let mut t=TrainingDashboard::new("T"); t.add_point(1,1.0);
        let h=full_dashboard("Full",&m,&t);
        assert!(h.contains("Architecture")&&h.contains("Metrics"));
    }
    #[test]
    fn launch_file() {
        let _=launch("<html>test</html>","/tmp/test_gui.html");
        assert!(std::path::Path::new("/tmp/test_gui.html").exists());
    }
}
