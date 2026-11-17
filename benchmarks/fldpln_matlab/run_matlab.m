function run_matlab(work, run)
% The MATLAB side of run_benchmark.py: load work/inputs.mat, warm up on one segment, then build the floodplain
% table of every segment in turn, writing the marker files <run>.loaded, <run>.ready and <run>.done the benchmark
% measures between. <run>.done holds the seconds the segments took.
load(fullfile(work, 'inputs.mat'), 'fildat', 'fdrdat', 'seg_info', 'segs', 'warmup', 'fldmn', 'fldmx', 'dh', ...
     'ssflg', 'bg');
inp.bildr = [work, filesep];
inp.fil = fullfile(work, 'fil.bil');  % read for its header only: the model takes fildat
inp.dem = inp.fil;                    % read for the fill depth of the fourth column
inp.seg = seg_info;
mark(work, run, 'loaded', '');

% The model skips a segment whose table is already saved, so each run saves its own
inp.segdr = [fullfile(work, [run, '_warmup']), filesep];
fldpln_model_v5ram(warmup, inp, fildat, fdrdat, fldmn, fldmx, dh, 0, ssflg, bg);
inp.segdr = [fullfile(work, run), filesep];
mark(work, run, 'ready', '');
start = tic;
for k = 1:numel(segs)
   fldpln_model_v5ram(segs(k), inp, fildat, fdrdat, fldmn, fldmx, dh, 0, ssflg, bg);
end
seconds = toc(start);
mark(work, run, 'done', sprintf('{"seconds": %.6f, "version": "%s"}', seconds, version));

% Every segment's rows in one table, its FSP, FPP, DTF and the segment's place in segs, for the benchmark to check
% (the model saves v7.3, which SciPy doesn't read). The model names its tables by the depths, as
% h<fldmx>_dh<dh>_seg<segment> with 'p' for '.'.
name = @(seg) sprintf('h%s_dh%s_seg%d', strrep(num2str(fldmx), '.', 'p'), strrep(num2str(dh), '.', 'p'), seg);
tables = cell(numel(segs), 1);
for k = 1:numel(segs)
   s = load([inp.segdr, name(segs(k)), '.mat'], 'fldpln');
   tables{k} = [s.fldpln(:, 1:3), repmat(k, size(s.fldpln, 1), 1)];
end
fldpln = vertcat(tables{:});
save(fullfile(work, [run, '_rows.mat']), 'fldpln', '-v7');
end

function mark(work, run, stage, text)
fid = fopen(fullfile(work, [run, '.', stage]), 'w');
fprintf(fid, '%s', text);
fclose(fid);
end
