abstract type AbstractAdaptor{PT <: Proposal} end
# TODO: Extend to arrays and named tuples

function initialize! end
function update! end

noadaptation(proposal::PT) where {PT} = NoAdaptation{PT}()
noadaptation(proposal::AbstractArray) = map(noadaptation, proposal)
noadaptation(proposal::NamedTuple) = map(noadaptation, proposal)

initialize!(adapt::AbstractArray, proposal::AbstractArray, init_params) = foreach(initialize!, adapt, proposal, init_params)
initialize!(adapt::NamedTuple, proposal::NamedTuple, init_params) = foreach(initialize!, adapt, proposal, init_params)

adapt!(adapt::AbstractArray, proposal::AbstractArray, params, accept::Union{Val{true},Val{false}}) =
    foreach((a, pr, p) -> adapt!(a, pr, p, accept), adapt, proposal, params)

adapt!(adapt::NamedTuple, proposal::NamedTuple, params, accept::Union{Val{true},Val{false}}) = 
    foreach((a, pr, p) -> adapt!(a, pr, p, accept), adapt, proposal, params)

struct NoAdaptation{PT} <: AbstractAdaptor{PT} end

initialize!(::NoAdaptation{PT}, proposal::PT, init_params) where {PT} = nothing
adapt!(::NoAdaptation{PT}, proposal::PT, params, accept::Union{Val{true},Val{false}}) where {PT} = nothing

# Simple Adaptive Metropol is Proposal
# The proposal at each step is equal to a scalar multiple
# of the empirical posterior covariance plus a fixed, small covariance
# matrix epsilon which is also used for initial exploration.
# 
# Reference:
#   H. Haario, E. Saksman, and J. Tamminen, "An adaptive Metropolis algorithm",
#   Bernoulli 7(2): 223-242 (2001)
mutable struct AMAdaptor{MNT <: AbstractMvNormal, 
                         FT <: AbstractFloat,
                         CT <: AbstractPDMat} <: AbstractAdaptor{SymmetricRandomWalkProposal{MNT}}
    epsilon::CT
    scalefactor::FT
    samplemean::Vector{FT}
    samplesqmean::Matrix{FT}
    center::Vector{FT}          # constant = zeros(length(samplemean)), to save allocations
    N::Int
end

function AMAdaptor(prop::SymmetricRandomWalkProposal, 
                   scalefactor=2.38^2 / length(prop.proposal))
    AMAdaptor(prop.proposal, scalefactor)
end

function AMAdaptor(prop::AbstractMvNormal,
                   scalefactor=2.38^2 / length(prop))
        AMAdaptor{typeof(prop),eltype(cov(prop)),typeof(prop.Σ)}(
                  prop.Σ,
                  scalefactor, 
                  zeros(length(prop)), 
                  zeros(length(prop), length(prop)), 
                  zeros(length(prop)), 0)
end

# When the proposal is initialised the empirical posterior covariance is zero
function initialize!(adapt::AMAdaptor{MNT}, 
                     proposal::SymmetricRandomWalkProposal{MNT}, params) where {MNT}
    adapt.samplemean .= params
    mul!(adapt.samplesqmean, params, params')
    adapt.N = 1
    proposal.proposal = MvNormal(adapt.center, deepcopy(adapt.epsilon))
end

# Recompute the empirical posterior covariance matrix
# TODO: Use Welford's algorithm for increased stability
function adapt!(adapt::AMAdaptor{MNT}, proposal::SymmetricRandomWalkProposal{MNT}, params, 
                accept::Union{Val{true}, Val{false}}) where {MNT}
    adapt.N += 1
    adapt.samplemean .+= (params .- adapt.samplemean) ./ adapt.N 
    adapt.samplesqmean .= (adapt.samplesqmean .* (adapt.N - 1) + params * params') ./ adapt.N

    prop_cov = adapt.scalefactor .* adapt.samplesqmean
    mul!(prop_cov, adapt.samplemean, adapt.samplemean', -adapt.scalefactor, 1.0)
    proposal.proposal = MvNormal(adapt.center, pdadd(prop_cov, adapt.epsilon))
end
