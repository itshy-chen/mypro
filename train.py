Y0, S0 = decoder.forward_initial(...)
f0 = future_encoder(Y0.detach())      # ← 这里做 detach，模块本身不做
nb = geometry.future_knn(Y0, K=16)
b_geom = geometry.geom_bias(Y0, nb)
