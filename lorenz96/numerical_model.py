import numpy as np

def lorenz96(t, x, F=8):
    """Lorenz 96 model with constant forcing"""
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return dxdt

function lorenz96_twoscale(t, u, N=40, n=5, F=8)
   dx = zeros(N)
   dy = zeros(n, N)

   u = reshape(u, n + 1, N)
   x = u[1, :]
   y = u[2:end, :]

   for i=1:N
      dx[i] = (x[mod(i+1, 1:N)] - x[mod(i-2, 1:N)])*x[mod(i-1, 1:N)] - x[i] + F - p["h"]*p["c"]/p["b"]*sum(y[:, i])

      for j=1:n
         if j == n
           jp1 = 1
           jp2 = 2
           jm1 = n - 1
           ip1 = mod(i + 1, 1:N)
           ip2 = mod(i + 1, 1:N)
           im1 = i
         elseif j == n - 1
           jp1 = n
           jp2 = 1
           jm1 = n - 2
           ip1 = i
           ip2 = mod(i + 1, 1:N)
           im1 = i
         elseif  j == 1
           jp1 = 2
           jp2 = 3
           jm1 = n
           ip1 = i
           ip2 = i
           im1 = mod(i - 1, 1:N)
         else
           jp1 = j + 1
           jp2 = j + 2
           jm1 = j - 1
           ip1 = ip2 = im1 = i
         end
         dy[j, i] = p["c"]*p["b"]*y[jp1, ip1]*(y[jm1, im1] - y[jp2, ip2]) - p["c"]*y[j, i] + p["h"]*p["c"]/p["b"]*x[i]
      end
   end

   du = vec([dx dy']')

   return du
end
