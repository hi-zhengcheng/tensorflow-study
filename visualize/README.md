## visualize tensorflow graph
1. Run script to create tensorflow graph data in tmp dir.
    ```
    python visualize_tf_graph.py
    ```

1. Start the tensorboard server:
    ```
    tensorboard --logdir=tmp
    ```

1. View the tensorflow graph in your web browser: http://127.0.0.1:6006 . In `Graphs` tab, you can view your tensorflow graph.

1. Finally, shutdown your tensorboard server.
