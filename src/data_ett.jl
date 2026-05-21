# ETT (Electricity Transformer Temperature) data loader.
#
# Reads ETTh1.csv from data/ETTh1.csv and extracts the columns
# HUFL, MUFL, LUFL, OT (D=4 outputs). Time is the d=1 input.
#
# The file is expected to be present locally (downloaded manually from
# https://github.com/zhouhaoyi/ETDataset). No network access here.

"""
    load_etth1(; N=500, csv_path="data/ETTh1.csv", cols=("HUFL","MUFL","LUFL","OT"))
        -> (; X, Y, col_names, ot_idx)

Load ETTh1 and subsample N evenly-spaced rows.

Returns:
- `X::Matrix{Float64}` — `N × 1` normalized time index (linspace on [-1, 1])
- `Y::Vector{Vector{Float64}}` — length-N vector of D-vectors (one per timestamp)
- `col_names::Vector{String}` — the D output column names in order
- `ot_idx::Int` — index of OT within `col_names`

Throws an informative error if the CSV is missing.
"""
function load_etth1(; N::Int=500,
                    csv_path::AbstractString=joinpath(@__DIR__, "..", "data", "ett", "ETTh1.csv"),
                    cols::NTuple{D, String}=("HUFL", "MUFL", "LUFL", "OT")) where D
    isfile(csv_path) || error("""
        ETTh1.csv not found at $(abspath(csv_path)).
        Download it manually:
          mkdir -p data/ett && curl -L -o data/ett/ETTh1.csv \\
            https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv
        """)

    lines = readlines(csv_path)
    header = split(lines[1], ',')
    col_idx = [findfirst(==(c), header) for c in cols]
    any(isnothing, col_idx) && error("Columns $(cols) not all found in header $(header)")
    col_idx = Int.(col_idx)

    n_total = length(lines) - 1
    N <= n_total || error("Requested N=$N but only $n_total rows available")

    # Evenly-spaced stride (deterministic, independent of seed)
    stride = collect(round.(Int, range(1, n_total, length=N)))

    Y = Vector{Vector{Float64}}(undef, N)
    for (i, row_idx) in enumerate(stride)
        fields = split(lines[1 + row_idx], ',')
        Y[i] = [parse(Float64, fields[j]) for j in col_idx]
    end

    # Normalized time index on [-1, 1]
    X = reshape(collect(range(-1.0, 1.0, length=N)), N, 1)

    ot_idx = findfirst(==("OT"), [string(c) for c in cols])
    (; X, Y, col_names=String[cols...], ot_idx=Int(ot_idx))
end
