idf_component_register(
    SRCS "app_main.cpp"
    INCLUDE_DIRS "."
    EMBED_FILES "snn_weights.bin" "replay_data.bin"
)