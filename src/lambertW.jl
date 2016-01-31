export lambertW

#
# The Lambert W function (See reference [4])
# The following algorithm computes the Lambert-W function to 15 correct digits using Halley's method for any x
# in the domain I=( -1/exp(1), ∞ ). This function can also compute the lambert-W function component-wise for
# vectors or matrices. The Lambert-W function is used in the module DESincEig to compute the mesh size h.
# Input: x:: Number, Vector or Matrix
# Output: lambertW(x)
#
function lambertW(x::Real)
    if x < -exp(-one(x))
        return throw(DomainError())
    elseif x < -0.2576872774
        w0 = -(5+2*e*x)/3 + sqrt(2*(e*x+one(x)))
    elseif x ≤ 1.272507835
        w0 = x-x^2
    else
        w0 = log(x)-log(log(x))
    end
    expw0 = exp(w0)
    w1 = w0 - (w0*expw0 - x)/((w0 + 1)*expw0 -
        (w0 + 2) * (w0*expw0 - x)/(2w0 + 2))
    while abs(w1/w0 - 1) > 2eps(typeof(x))
        w0 = w1
        expw0 = exp(w0)
        w1 = w0 - (w0*expw0 - x)/((w0 + 1)*expw0 -
            (w0 + 2) * (w0*expw0 - x)/(2w0 + 2))
    end
    return w1
end
lambertW(x::Integer) = lambertW(float(x))
@vectorize_1arg Real lambertW
