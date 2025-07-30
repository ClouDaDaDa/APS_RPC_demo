import json
import socket
import random
import time
from time import strftime
from threading import Thread, active_count


class Schemer:
    """Simple demo scheduler class"""
    
    def __init__(self):
        self.request_count = 0
        self.algorithms = {
            'est_eet': 'Earliest Start Time - Earliest End Time',
            'spt': 'Shortest Processing Time',
            'random': 'Random Selection'
        }
    
    def scheme(self, key1, key2):
        """
        Execute scheduling algorithm
        
        Args:
            key1: Algorithm type (est_eet, spt, random)
            key2: Task data or parameters
        Returns:
            dict: Scheduling result
        """
        self.request_count += 1
        
        # Simulate processing time
        time.sleep(0.1)
        
        # Execute different scheduling logic based on algorithm type
        if key1 == 'est_eet':
            result = self._est_eet_algorithm(key2)
        elif key1 == 'spt':
            result = self._spt_algorithm(key2)
        elif key1 == 'random':
            result = self._random_algorithm(key2)
        else:
            result = self._default_algorithm(key2)
        
        return {
            'algorithm': key1,
            'input_data': key2,
            'result': result,
            'request_id': self.request_count,
            'timestamp': strftime("%Y-%m-%d %H:%M:%S"),
            'status': 'success'
        }
    
    def _est_eet_algorithm(self, data):
        """EST-EET algorithm demo"""
        # Simulate workstation and machine data
        workstations = data.get('workstations', 3)
        machines = data.get('machines', 5)
        jobs = data.get('jobs', 10)
        
        # Simulate scheduling calculation
        makespan = random.randint(50, 200)
        total_time = random.randint(100, 500)
        
        return {
            'makespan': makespan,
            'total_time': total_time,
            'workstations_used': workstations,
            'machines_used': machines,
            'jobs_scheduled': jobs,
            'efficiency': round(random.uniform(0.7, 0.95), 3)
        }
    
    def _spt_algorithm(self, data):
        """SPT algorithm demo"""
        jobs = data.get('jobs', 8)
        priority = data.get('priority', 'normal')
        
        # Simulate SPT scheduling
        processing_time = random.randint(30, 150)
        completion_time = random.randint(60, 300)
        
        return {
            'processing_time': processing_time,
            'completion_time': completion_time,
            'jobs_processed': jobs,
            'priority_level': priority,
            'utilization': round(random.uniform(0.6, 0.9), 3)
        }
    
    def _random_algorithm(self, data):
        """Random algorithm demo"""
        tasks = data.get('tasks', 6)
        
        # Simulate random scheduling result
        return {
            'tasks_assigned': tasks,
            'random_score': round(random.uniform(0.1, 1.0), 3),
            'distribution': 'random',
            'performance': round(random.uniform(0.5, 0.8), 3)
        }
    
    def _default_algorithm(self, data):
        """Default algorithm demo"""
        return {
            'message': 'Unknown algorithm, using default',
            'data_processed': data,
            'default_result': 'demo_scheduling_completed'
        }


def tcp_server(host='127.0.0.1', port=23333):
   with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
      s.bind((host, port))
      s.listen(64)
      while True:
         conn, addr = s.accept()
         t = Thread(target=plot, args=(conn, addr))
         t.start()


def plot(conn, addr):
   print(strftime("%m-%d %H:%M:%S"), '%s:%s connected, active clients:' % addr, active_count()-1)
   schemer = Schemer()
   with conn:
      while True:
         data = ''
         while True:
            try:
               piece = conn.recv(8192)
               if not piece:
                  raise Exception("Client quit.")
            except:
               print(strftime("%m-%d %H:%M:%S"), '%s:%s closed, active clients:' % addr, active_count()-2)
               return
            piece = piece.decode('utf-8')
            if piece[0]=='<':
               data = piece
            else:
               data += piece
            if data[-1]=='>':
               break
         data = json.loads(data[1:-1])
         data = schemer.scheme(data['key1'], data['key2'])
         data = '<'+json.dumps(data, separators=(',',':'))+'>'
         conn.sendall(data.encode('utf-8'))


if __name__ == '__main__':
   tcp_server()
