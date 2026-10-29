function run_segments(work)
% Run fldpln_model_v5ram on each segment of work/inputs.mat with each spillover flag in ssflgs, and save each table
% as work/matlab_<segment>_ss<ssflg>.mat in a MAT format SciPy reads (the model saves v7.3, which it doesn't).
load(fullfile(work, 'inputs.mat'), 'fildat', 'fdrdat', 'seg_info', 'segs', 'fldmn', 'fldmx', 'dh', 'ssflgs', 'bg');
inp.bildr = [work, filesep];
inp.fil = fullfile(work, 'fil.bil');  % read for its header only: the model takes fildat
inp.dem = inp.fil;                    % read for the fill depth of the fourth column
inp.seg = seg_info;
for ssflg = ssflgs(:)'
   % The model skips a segment whose output is already there, and doesn't name it by ssflg: a folder per flag
   inp.segdr = [fullfile(work, sprintf('segment_files_ss%d', ssflg)), filesep];
   for k = 1:numel(segs)
      fldpln_model_v5ram(segs(k), inp, fildat, fdrdat, fldmn, fldmx, dh, 0, ssflg, bg);
      % The model names its output by the depths, as h<fldmx>_dh<dh>_seg<segment> with 'p' for '.'
      outf = sprintf('h%s_dh%s_seg%d', strrep(num2str(fldmx), '.', 'p'), strrep(num2str(dh), '.', 'p'), segs(k));
      s = load([inp.segdr, outf, '.mat'], 'fldpln');
      fldpln = s.fldpln;
      save(fullfile(work, sprintf('matlab_%d_ss%d.mat', segs(k), ssflg)), 'fldpln', '-v7');
   end
end
end
