#####################
#   MODEL
#####################

struct Joint{Tℓprior, Tℓll} <: Function
    ℓprior      :: Tℓprior
    ℓlikelihood :: Tℓll
end

function (joint::Joint)(θ)
    return joint.ℓprior(θ) .+ joint.ℓlikelihood(θ)
end


struct TemperedJoint{Tℓprior, Tℓll, T<:AbstractFloat} <: Function
    ℓprior      :: Tℓprior
    ℓlikelihood :: Tℓll
    β           :: T
end

function (tj::TemperedJoint)(θ)
    return tj.ℓprior(θ) .+ (tj.ℓlikelihood(θ) .* tj.β)
end


function MCMCTempering.make_tempered_model(model::DensityModel, β::T) where {T<:AbstractFloat}
    logdensity_β = TemperedJoint(model.logdensity.ℓprior, model.logdensity.ℓlikelihood, β)
    model_β = DensityModel(logdensity_β)
    return model_β
end



#####################
#   SWAPPING
#####################

function MCMCTempering.make_tempered_logπ(model::DensityModel, β::T) where {T<:AbstractFloat}
    function logπ(z)
        return logdensity(model, z) * β
    end
    return logπ
end


function MCMCTempering.get_θ(trans::Transition)
    return trans.params
end
