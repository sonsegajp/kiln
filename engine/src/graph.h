// Node-graph executor: runs ComfyUI API-format workflows natively (see docs/GRAPH_PROTOCOL.md).
#pragma once
#include <string>

#include "engine.h"

// The "graph" command. `req` is the whole request line; events are emitted as the graph runs.
void run_graph(Engine& E, const Json& req);
