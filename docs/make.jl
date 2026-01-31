using SpikeAnalysis
using Documenter

DocMeta.setdocmeta!(SpikeAnalysis, :DocTestSetup, :(using SpikeAnalysis); recursive=true)

makedocs(;
    modules=[SpikeAnalysis],
    authors="Galen Lynch <galen@galenlynch.com>",
    sitename="SpikeAnalysis.jl",
    format=Documenter.HTML(;
        canonical="https://galenlynch.github.io/SpikeAnalysis.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/galenlynch/SpikeAnalysis.jl",
    devbranch="main",
)
