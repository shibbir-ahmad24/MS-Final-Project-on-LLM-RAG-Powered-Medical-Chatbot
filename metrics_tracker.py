import time
import json
import os

class MetricsTracker:
    def __init__(self):
        self.total_queries = 0
        self.successful_routings = 0
        self.failed_routings = 0
        self.response_times = []

    def record_query(self, routed_correctly: bool, response_time: float):
        self.total_queries += 1
        if routed_correctly:
            self.successful_routings += 1
        else:
            self.failed_routings += 1
        self.response_times.append(response_time)

    def get_metrics_summary(self):
        if self.total_queries == 0:
            accuracy = 0.0
            avg_response_time = 0.0
        else:
            accuracy = (self.successful_routings / self.total_queries) * 100
            avg_response_time = sum(self.response_times) / self.total_queries

        return {
            "Total Queries": self.total_queries,
            "Successful Routings": self.successful_routings,
            "Failed Routings": self.failed_routings,
            "Routing Accuracy (%)": round(accuracy, 2),
            "Average Response Time (sec)": round(avg_response_time, 2)
        }

    def print_metrics_summary(self):
        summary = self.get_metrics_summary()
        print("\n=== Metrics Summary ===")
        for k, v in summary.items():
            print(f"{k}: {v}")
        print("==========================\n")
