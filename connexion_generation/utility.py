from scipy import sparse


def switch_connexion(W, i, j, value):
    W = sparse.lil_matrix(W)
    W[i, j] = value
    W = sparse.coo_matrix(W)
    if value == 0:
        W.eliminate_zeros()
    return W


def change_connexion(W, i, j, value):
    # i for rows, j for columns
    W = sparse.lil_matrix(W)
    W[i, j] = W[i, j] + value
    if W[i, j] < 0:
        W[i, j] = 0
    W = sparse.coo_matrix(W)
    W.eliminate_zeros()
    return W

#
# def compare(W_old, W_new, Win, bias, Wout, activation_function, b_out, U_test, y_test, last_state, leaky_rate=1):
#     y_pred_old = run(W_old, Win, bias, Wout, U_test, activation_function, b_out, last_state, leaky_rate)
#     nrmse_old = nrmse(y_test, y_pred_old)
#
#     y_pred_new = run(W_new, Win, bias, Wout, U_test, activation_function, b_out, last_state)
#     nrmse_new = nrmse(y_test, y_pred_new)
#     if float(nrmse_new) < float(nrmse_old):
#         print(float(nrmse_new))
#         return W_new, 0
#     else:
#         return W_old, 1
