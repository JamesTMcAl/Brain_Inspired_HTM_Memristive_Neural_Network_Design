%%written by Jan Simon
function Hash = DataHash(Data, Opt)
% DATAHASH  Create a hash value for any array (numeric, logical, char, struct, cell …)
%
%   Hash = DataHash(Data)           - returns lowercase MD5 hex string
%   Hash = DataHash(Data, Opt)      - see below for options
%
% Basic idea:
%   1. Serialize the input recursively into a uint8 byte stream
%   2. Run the stream through java.security.MessageDigest
%
% Minimal option struct:
%   Opt.Method   : 'MD5' (default) | 'SHA-1' | 'SHA-256' …
%   Opt.Format   : 'hex' (default) | 'uint8'
%
% Compatible with Jan Simon’s original function for the arguments used
% in compute_overlap.m (single input, default options).

% ---------- default options ----------
if nargin < 2 || ~isstruct(Opt)
    Opt = struct;
end
if ~isfield(Opt,'Method')
    Opt.Method = 'MD5';
end
if ~isfield(Opt,'Format')
    Opt.Format = 'hex';
end

% ---------- serialize the data ----------
% — guard Java & MATLAB version
if ~usejava('jvm')
    error('DataHash:NoJava','DataHash requires Java; JVM is disabled.');
end
if verLessThan('matlab','7.6')  % R2008a
    error('DataHash:OldMATLAB','DataHash requires MATLAB ≥ R2008a.');
end
% ---------- serialize the data ----------
byteStream = getByteStreamFromArray(Data);   % built-in, R2008a+

% ---------- compute digest ----------
md     = java.security.MessageDigest.getInstance(upper(Opt.Method));
hash16 = typecast(md.digest(byteStream), 'uint8');   % big‑endian order

% ---------- return in requested format ----------
switch lower(Opt.Format)
    case 'hex'
        Hash = sprintf('%.2x', hash16);
    case 'uint8'
        Hash = reshape(hash16, 1, []);
    otherwise
        error('DataHash:BadFormat','Unknown Opt.Format value.');
end
end
