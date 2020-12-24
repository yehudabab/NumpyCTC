import numpy as np


class Alphabet:
    blank_label = '^'
    pure_alphabet = ['a', 'b', 'c', 'd']
    alphabet_letter_to_ind = {ch: ind for ind, ch in enumerate(pure_alphabet + [blank_label])}
    alphabet_ind_to_letter = {ind: ch for ind, ch in enumerate(pure_alphabet + [blank_label])}
    blank_ind = alphabet_letter_to_ind[blank_label]


def are_equal(f1, f2):
    return np.isclose(f1, f2)


def print_test(empirical_value, expected_value, test_name):
    message = 'OK' if are_equal(empirical_value, expected_value) else 'FAIL'
    print('[Test %s] %s check: gt score: %.6f empirical score: %.6f' %
          (message, test_name, expected_value, empirical_value))


def pad_label(label):
    return '^%s^' % '^'.join(label)


def create_alpha_beta(gt_label, outputs):
    padded_gt_label = pad_label(gt_label)  # l' from the paper. gt_label is l from the paper
    num_time_steps = outputs.shape[0]
    padded_gt_label_length = len(padded_gt_label)
    last_padded_ind = padded_gt_label_length - 1
    blank_label = Alphabet.blank_label

    # To avoid expensive recursion, we use dynamic programming to fill tables of size (T, |l'|) for alpha, beta.

    # Alpha:
    alpha_table = np.zeros((num_time_steps, padded_gt_label_length))

    def alpha(t, s):
        if s < 0 or s >= len(padded_gt_label):
            return 0

        current_padded_character = padded_gt_label[s]
        current_padded_label_score = outputs[t, Alphabet.alphabet_letter_to_ind[current_padded_character]]

        if t == 0:
            if s == 0:
                return outputs[0, Alphabet.blank_ind]
            elif s == 1:
                return current_padded_label_score
            else:
                return 0

        # (6, 7) from the paper. No need to call alpha for previous time steps, because it was already calculated
        alpha_tag_t_s = alpha_table[t - 1, s] + (alpha_table[t - 1, s - 1] if s-1 >= 0 else 0)
        if current_padded_character == blank_label or (s >= 2 and padded_gt_label[s-2] == current_padded_character):
            return alpha_tag_t_s * current_padded_label_score
        else:
            return (alpha_tag_t_s + (alpha_table[t - 1, s - 2] if s - 2 >= 0 else 0)) * current_padded_label_score

    for t in range(0, num_time_steps):
        for s in range(0, padded_gt_label_length):
            alpha_table[t, s] = alpha(t, s)

    # Beta:
    beta_table = np.zeros((num_time_steps, padded_gt_label_length))

    def beta(t, s):
        if s < 0 or s >= len(padded_gt_label):
            return 0

        current_padded_character = padded_gt_label[s]
        current_padded_label_score = outputs[t, Alphabet.alphabet_letter_to_ind[current_padded_character]]
        last_time_step = outputs.shape[0] - 1

        if t == last_time_step:
            if s == last_padded_ind:
                return outputs[last_time_step, Alphabet.blank_ind]
            elif s == last_padded_ind - 1:
                return current_padded_label_score
            else:
                return 0

        # (10, 11) from the paper. No need to call beta for previous time steps, because it was already calculated
        beta_tag_t_s = beta_table[t + 1, s] + (beta_table[t + 1, s + 1] if s + 1 <= last_padded_ind else 0)
        if current_padded_character == blank_label or \
                (s + 2 <= last_padded_ind and padded_gt_label[s+2] == current_padded_character):
            return beta_tag_t_s * current_padded_label_score
        else:
            return (beta_tag_t_s +
                    (beta_table[t + 1, s + 2] if s + 2 <= last_padded_ind else 0)) * current_padded_label_score

    for t in range(num_time_steps - 1, -1, -1):
        for s in range(padded_gt_label_length - 1, -1, -1):
            beta_table[t, s] = beta(t, s)

    return alpha_table, beta_table


def generate_random_ctc_table(num_time_steps):
    alphabet_tag_size = len(Alphabet.alphabet_letter_to_ind)
    return np.random.rand(num_time_steps, alphabet_tag_size)


def test_alpha_beta(outputs):
    label = ''.join([Alphabet.pure_alphabet[t % len(Alphabet.pure_alphabet)] for t in range(outputs.shape[0])])
    alpha_dp_table, beta_dp_table = create_alpha_beta(label, outputs)
    last_time_step_ind = outputs.shape[0] - 1
    padded_label_length = 2 * len(label) + 1
    label_score_manually = np.prod([outputs[t, Alphabet.alphabet_letter_to_ind[label[t]]] for t in range(outputs.shape[0])])

    # formula (8) from the paper
    score_last = alpha_dp_table[last_time_step_ind, padded_label_length - 1]
    score_before_last = alpha_dp_table[last_time_step_ind, padded_label_length - 2]
    label_score_by_alpha = score_last + score_before_last

    # similarly to formula (8), P(label|outputs) is the sum of the probabilities of l' with and without the
    # first blank at time step 0:
    score_first = beta_dp_table[0, 0]
    score_second = beta_dp_table[0, 1]
    label_score_by_beta = score_first + score_second

    print('> Alpha, Beta Tests')
    print('\t CTC')
    print(outputs.T)
    print('\n\t GT Label: \'%s\'' % label)
    print('\n\t Tests')
    print_test(label_score_by_alpha, label_score_manually, 'Alpha DP')
    print_test(label_score_by_beta, label_score_manually, 'Beta DP')

    # Check formula 14
    padded_label = pad_label(label)
    for t in range(outputs.shape[0]):
        prob_sum = 0.
        for s in range(padded_label_length):
            padded_char_at_s = padded_label[s]
            score_at_s_t = outputs[t, Alphabet.alphabet_letter_to_ind[padded_char_at_s]]
            alpha_beta_prod_at_t_s = alpha_dp_table[t, s] * beta_dp_table[t, s]
            prob_sum += alpha_beta_prod_at_t_s / score_at_s_t
        print_test(prob_sum, label_score_manually, 'Formula18 @ t=%i' % t)

    print('\n***************************\n')


def calculate_gradients_for_ctc_layer(outputs, gt_label):
    assert outputs.shape[0] >= len(gt_label)
    alpha_dp_table, beta_dp_table = create_alpha_beta(gt_label, outputs)

    padded_gt_label = pad_label(gt_label)
    gradients = np.zeros_like(outputs)

    score_last = alpha_dp_table[outputs.shape[0] - 1, len(padded_gt_label) - 1]
    score_before_last = alpha_dp_table[outputs.shape[0] - 1, len(padded_gt_label) - 2]
    p_l_given_ctc = score_last + score_before_last

    for t in range(outputs.shape[0]):
        for k in range(outputs.shape[1]):

            # Formula 15:
            d_p_d_ytk = 0
            lab_lk = np.nonzero(
                list(map(lambda x: 1 if Alphabet.alphabet_ind_to_letter[k] in x else 0, padded_gt_label)))[0]
            for s in lab_lk:
                d_p_d_ytk += alpha_dp_table[t, s] * beta_dp_table[t, s]

            d_p_d_ytk /= (outputs[t, k] ** 2)
            d_lnp_d_ytk = (1. / p_l_given_ctc) * d_p_d_ytk
            gradients[t, k] = d_lnp_d_ytk
    return gradients


def grads_print(ctc_table):
    label = 'bab'
    print('> Calculating gradients of the CTC matrix')
    ctc_table_grad = calculate_gradients_for_ctc_layer(ctc_table, label)
    print('\t CTC')
    print(ctc_table.T)
    print('\n\t GT Label: \'%s\'' % label)
    print('\n\t CTC Gradients')
    print(ctc_table_grad.T)


if __name__ == '__main__':
    ctc_table = generate_random_ctc_table(num_time_steps=6)
    test_alpha_beta(ctc_table)
    grads_print(ctc_table)

