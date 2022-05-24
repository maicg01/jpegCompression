import matplotlib.pyplot as plt
plt.xlim(10,40)
plt.ylim(10,40)

plt.plot([11.73,18.96,24.77,30.01,34.57], [29.714,27.943,26.796,25.885,25.122], "ro-")

plt.title('PSRN vs Compression Ratio')
# plt.title đặt tên cho biểu đồ

plt.xlabel('Compression Ratio')
plt.ylabel('PSRN')
# plt.xlabel và plt.ylabel đặt tên trục x,y cho biểu đồ

# plt.legend(loc='best') #thêm chú thích
plt.show() # hiển thị biểu đồ



# import matplotlib.pyplot as plt

# plt.plot([3000,10000,30000,100000,300000], [10,20,30,40,50], color='blue', label='Selection Sort')
# plt.plot([3000,10000,30000,100000,300000], [15,25,35,45,55],color='green', label='Merge Sort')
# plt.plot([3000,10000,30000,100000,300000], [20,30,40,50,60],color='red', label='Heap Sort')
# plt.plot([3000,10000,30000,100000,300000], [25,35,45,55,65],color='black', label='Quick Sort')
# # [3000,10000,30000,100000,300000], [10,20,30,40,50] là các mảng tọa độ x,y
# # color hiển thị màu đường đồ thị
# # label Chú thích đồ thi

# plt.title('Chạy với dữ liệu sắp xếp giảm dần')
# # plt.title đặt tên cho biểu đồ

# plt.xlabel('Số dữ liệu đầu vào')
# plt.ylabel('Thời gian')
# # plt.xlabel và plt.ylabel đặt tên trục x,y cho biểu đồ

# plt.legend(loc='best') #thêm chú thích
# plt.show() # hiển thị biểu đồ