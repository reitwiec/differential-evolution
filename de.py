H, W = 32, 32

def visualize_attack(original_img, delta, target_class=None):
  adv_img_tensor = (original_img + delta).detach()
  orig_img = unnormalize(original_img.cpu().squeeze()).permute(1, 2, 0).numpy()
  adv_img = unnormalize(adv_img_tensor.cpu().squeeze()).permute(1, 2, 0).numpy()

  # normalizing to [0,1]
  delta_vis = delta.detach().cpu().squeeze().numpy()  # (C, H, W)
  delta_vis = (delta_vis - delta_vis.min()) / (delta_vis.max() - delta_vis.min() + 1e-8)
  delta_vis = np.transpose(delta_vis, (1, 2, 0))  # H x W x C

  with torch.no_grad():
    orig_logits = cnn(original_img.unsqueeze(0) if original_img.dim() == 3 else original_img)
    adv_logits = cnn(adv_img_tensor)

    delta_normalized = torch.from_numpy(np.transpose(delta_vis, (2,0,1))).unsqueeze(0).float().to(device)
    delta_logits = cnn(renormalize(delta_normalized))

    orig_probs = F.softmax(orig_logits, dim=1).squeeze(0).cpu().numpy()
    adv_probs = F.softmax(adv_logits, dim=1).squeeze(0).cpu().numpy()
    delta_probs = F.softmax(delta_logits, dim=1).squeeze(0).cpu().numpy()



  orig_cls, orig_conf = top1(orig_probs)
  adv_cls, adv_conf = top1(adv_probs)
  delta_cls, delta_conf = top1(delta_probs)

  # Plot results
  fig, axes = plt.subplots(1, 3, figsize=(12, 4))

  axes[0].imshow(orig_img)
  axes[0].axis('off')
  axes[0].set_title(f"Original\n{orig_cls} ({orig_conf:.1f}%)")

  axes[1].imshow(delta_vis)
  axes[1].axis('off')
  axes[1].set_title(f"Perturbation\n{delta_cls} ({delta_conf:.1f}%)")

  axes[2].imshow(adv_img)
  axes[2].axis('off')
  title_color = 'red' if adv_cls != orig_cls else 'green'
  axes[2].set_title(f"Adversarial\n{adv_cls} ({adv_conf:.1f}%)", color=title_color)

  plt.tight_layout()
  plt.show()

  if target_class is None:
    attack_success = adv_cls != orig_cls
    print(f"UNTARGETED ATTACK")
    print(f"Attack successful: {attack_success} (changed prediction: {orig_cls} → {adv_cls})")
  else:
    attack_success = adv_cls == target_class
    print(f"TARGETED ATTACK (target: {target_class})")
    print(f"Attack successful: {attack_success} ({orig_cls} → {adv_cls})")
    if attack_success:
      print(f"Successfully fooled model to predict '{target_class}'")
    else:
      print(f"Failed to reach target '{target_class}', got '{adv_cls}' instead")


def random_candidate():
  #random candidate [x,y,r,g,b]
  return np.array([
        random.uniform(0, W-1),
        random.uniform(0, H-1),
        random.uniform(0, 1),
        random.uniform(0, 1),
        random.uniform(0, 1)
    ], dtype=float)
  
def decode(vec):
  #converting candidate into pixel and rgb
  x = int(round(vec[0]))
  y = int(round(vec[1]))
  x = max(0, min(W-1, x))
  y = max(0, min(H-1, y))
  r = float(np.clip(vec[2], 0, 1))
  g = float(np.clip(vec[3], 0, 1))
  b = float(np.clip(vec[4], 0, 1))
  return x, y, (r,g,b)

def apply_one_pixel(base_img_norm, candidate_vec):
  x,y,(r,g,b) = decode(candidate_vec)
  raw = unnormalize(base_img_norm[0])
  raw = raw.clone()
  raw[0,y,x] = r
  raw[1,y,x] = g
  raw[2,y,x] = b
  norm = renormalize(raw)              # back to normalized
  return norm.unsqueeze(0)

def candidate_fitness(candidate_vec, base_img_norm, true_label, model, device):
  adv = apply_one_pixel(base_img_norm, candidate_vec).to(device)
  model.eval()
  with torch.no_grad():
      logits = model(adv)
      loss = criterion(logits, torch.tensor([true_label], device=device))
  return float(loss.item())


def de_one_pixel(base_img_norm, true_label, model, device, pop=400, gens=75, F=0.5):
  population = [random_candidate() for _ in range(pop)]
  fitnesses = [candidate_fitness(candidate, base_img_norm, true_label, model, device)
                 for candidate in population]
  evals = pop

  best_idx = int(np.argmax(fitnesses))
  best_vec = population[best_idx].copy()
  best_fit = fitnesses[best_idx] #fitness of the current best candidate

  history = [(evals, best_fit)]

  #if the image is already misclassified return as it is
  with torch.no_grad():
      pred0 = model(base_img_norm.to(device)).argmax(dim=1).item()
  if pred0 != true_label:
      return True, best_vec, best_fit, evals, history

  for gen in range(gens):
    for i in range(pop):
      #mutation
      idxs = list(range(pop))
      #remove the current candidate and then find 3 random candidates
      idxs.remove(i)
      a,b,c = random.sample(idxs, 3)
      A,B,C = population[a], population[b], population[c]
      mutant = A + F * (B - C)

      trial = mutant
      trial_fit = candidate_fitness(trial, base_img_norm, true_label, model, device)
      evals +=1

      # selection
      if trial_fit > fitnesses[i]:
        population[i] = trial
        fitnesses[i] = trial_fit
        if trial_fit > best_fit:
          best_fit = trial_fit
          best_vec = trial.copy()

      history.append((evals, best_fit))

      # check success
      adv = apply_one_pixel(base_img_norm, best_vec).to(device)
      with torch.no_grad():
        pred_adv = model(adv).argmax(dim=1).item()
      if pred_adv != true_label:
        return True, best_vec, best_fit, evals, history

  # final check
  adv = apply_one_pixel(base_img_norm, best_vec).to(device)
  with torch.no_grad():
    pred_adv = model(adv).argmax(dim=1).item()
  success = (pred_adv != true_label)
  return success, best_vec, best_fit, evals, history


def apply_k_pixels(base_img_norm, pixel_vecs):
    img = base_img_norm.clone()
    for vec in pixel_vecs:
        img = apply_one_pixel(img, vec)
    return img

def k_pixel_attack(base_img_norm, true_label, model, device,
                           K):
    pixel_list = []
    evals_total = 0
    adv = base_img_norm.clone()
    history = []

    with torch.no_grad():
        pred0 = model(adv.to(device)).argmax(dim=1).item()
    if pred0 != int(true_label):
        return True, pixel_list, adv, evals_total, [(0, 0.0)]

    for k in range(K):
        success, best_vec, best_fit, evals, hist = de_one_pixel(
            base_img_norm=adv,
            true_label=int(true_label),
            model=model,
            device=device
        )
        for (local_eval, local_fit) in hist:
            history.append((evals_total + local_eval, local_fit))
        evals_total += evals
        # store and apply
        pixel_list.append(best_vec)
        adv = apply_one_pixel(adv, best_vec)

    with torch.no_grad():
        pred_final = model(adv.to(device)).argmax(dim=1).item()
    return (pred_final != int(true_label)), pixel_list, adv, evals_total, history
  
success1, best_vec1, best_fit1, evals1, history1 = de_one_pixel(
    base_img_norm=ship_img,
    true_label=ship_label,
    model=cnn,
    device=device
)

success10, best_vec10, best_fit10, evals10, history10 = k_pixel_attack(
    base_img_norm=ship_img,
    true_label=ship_label,
    model=cnn,
    device=device,
    K=10
)

adv_norm1 = apply_one_pixel(ship_img, best_vec1)
delta1 = adv_norm1 - ship_img
visualize_attack(ship_img, delta1)

adv_norm10 = apply_k_pixels(ship_img, best_vec10)
delta10 = adv_norm10 - ship_img
visualize_attack(ship_img, delta10)