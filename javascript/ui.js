function ask_for_sliders_name(sd_model_checkpoint, dummy_component, low_instance_images, high_instance_images, _, train_mode_choose, target_prompt, positive_prompt, neutral_prompt, unconditional_prompt, resolution, checkpointing_steps, max_train_steps, train_batch_size, learning_rate, rank, network_alpha) {
    var name_ = prompt('Sliders id:');
    return [sd_model_checkpoint, dummy_component, low_instance_images, high_instance_images, name_, train_mode_choose, target_prompt, positive_prompt, neutral_prompt, unconditional_prompt, resolution, checkpointing_steps, max_train_steps, train_batch_size, learning_rate, rank, network_alpha];
}
